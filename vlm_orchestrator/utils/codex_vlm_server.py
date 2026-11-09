"""Loopback OpenAI chat-completions bridge to an authenticated Codex App Server.

Each robot request uses an ephemeral Codex thread. A serialized persistent
App Server process preserves its authenticated transport between requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from aiohttp import web

from vlm_orchestrator.vlm import parse_json


def convert_messages(messages: Any) -> list[dict]:
    """Preserve message order and every image in the App Server turn input."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    inputs = [
        {
            "type": "text",
            "text": (
                "You are the visual decision service for a robot simulator. Follow the "
                "task and response format supplied below. Inspect attached images "
                "directly. Return only the requested JSON object. Do not use tools."
            ),
        }
    ]
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            raise ValueError("unsupported message role")
        inputs.append({"type": "text", "text": f"[{message['role']} message]"})
        content = message.get("content")
        if isinstance(content, str):
            inputs.append({"type": "text", "text": content})
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be text or content parts")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("content part must be an object")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                inputs.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if (
                    not isinstance(url, str)
                    or not url.startswith("data:image/")
                    or ";base64," not in url
                ):
                    raise ValueError("image_url must contain a base64 image data URL")
                inputs.append({"type": "image", "url": url})
            else:
                raise ValueError("unsupported content part")
    return inputs


class AppServer:
    """Synchronous JSON-RPC transport, called under the HTTP service lock."""

    def __init__(
        self, model: str, effort: str, log_dir: Path, executable: str = "codex"
    ):
        self.model = model
        self.effort = effort
        self.log_dir = log_dir
        self.executable = executable
        self.process = None
        self.messages = Queue()
        self.sequence = 0
        self.thread_id = None
        self.turn_id = None
        self.stderr = None

    def _send(self, method, params, request_id=None):
        message = {"method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Codex App Server process is unavailable")
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    @staticmethod
    def _read(process, messages):
        try:
            for line in process.stdout:
                messages.put(json.loads(line))
        except (ValueError, OSError) as error:
            messages.put(error)
        finally:
            messages.put(None)

    def _receive(self, deadline, audit):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex turn timed out")
        try:
            message = self.messages.get(timeout=remaining)
        except Empty as error:
            raise TimeoutError("Codex turn timed out") from error
        if message is None or isinstance(message, Exception):
            raise RuntimeError("Codex App Server closed its output")
        audit.write(json.dumps(message, ensure_ascii=False) + "\n")
        audit.flush()
        if "method" in message and "id" in message:
            raise RuntimeError(f"Unexpected Codex server request: {message['method']}")
        return message

    def _request(self, method, params, deadline, audit):
        self.sequence += 1
        request_id = self.sequence
        self._send(method, params, request_id)
        while True:
            message = self._receive(deadline, audit)
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(str(message["error"]))
                return message["result"]

    def _start(self, deadline, audit):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError("Codex executable was not found")
        self.messages = Queue()
        self.stderr = (self.log_dir / "app-server.stderr.log").open(
            "a", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            [
                executable,
                "-a",
                "never",
                "-c",
                "model_providers.openai.supports_websockets=false",
                "app-server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=self.log_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        Thread(
            target=self._read, args=(self.process, self.messages), daemon=True
        ).start()
        self._request(
            "initialize",
            {"clientInfo": {"name": "voloagent_vlm", "version": "0.1.0"}},
            deadline,
            audit,
        )
        self._send("initialized", {})

    def run(self, payload, request_dir, timeout):
        deadline = time.monotonic() + timeout
        inputs = convert_messages(payload["messages"])
        self.thread_id = self.turn_id = None
        with (request_dir / "app-server-events.jsonl").open(
            "w", encoding="utf-8"
        ) as audit:
            try:
                self._start(deadline, audit)
                started = self._request(
                    "thread/start",
                    {
                        "model": self.model,
                        "ephemeral": True,
                        "sandbox": "read-only",
                        "approvalPolicy": "never",
                        "cwd": str(self.log_dir),
                    },
                    deadline,
                    audit,
                )
                self.thread_id = started["thread"]["id"]
                model = started.get("model", self.model)
                turn = self._request(
                    "turn/start",
                    {
                        "threadId": self.thread_id,
                        "input": inputs,
                        "effort": self.effort,
                    },
                    deadline,
                    audit,
                )
                self.turn_id = turn["turn"]["id"]
                final = None
                usage = None
                while True:
                    message = self._receive(deadline, audit)
                    params = message.get("params", {})
                    if params.get("threadId", self.thread_id) != self.thread_id:
                        continue
                    if message.get("method") == "thread/tokenUsage/updated":
                        usage = params.get("tokenUsage")
                    if message.get("method") == "item/completed":
                        item = params["item"]
                        if item["type"] == "agentMessage":
                            final = item["text"]
                        elif item["type"] not in {
                            "userMessage",
                            "reasoning",
                            "contextCompaction",
                        }:
                            raise RuntimeError(
                                f"Unexpected Codex activity: {item['type']}"
                            )
                    if message.get("method") == "turn/completed":
                        completed = params["turn"]
                        if completed["status"] != "completed":
                            raise RuntimeError(
                                str(completed.get("error") or completed["status"])
                            )
                        break
                if not isinstance(final, str):
                    raise RuntimeError("Codex completed without a final message")
                parsed = parse_json(final)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Codex response must be a JSON object")
                return {
                    "text": json.dumps(parsed, ensure_ascii=False),
                    "raw_text": final,
                    "thread_id": self.thread_id,
                    "turn_id": self.turn_id,
                    "model": model,
                    "usage": usage,
                }
            except Exception:
                self.close()
                raise

    def close(self):
        process = self.process
        if process is not None and process.poll() is None:
            if self.thread_id and self.turn_id:
                try:
                    self._send(
                        "turn/interrupt",
                        {"threadId": self.thread_id, "turnId": self.turn_id},
                    )
                except (OSError, RuntimeError, ValueError):
                    pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        self.process = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None


def build_app(
    *, model="gpt-6.1-sol", effort="low", log_dir: Path, timeout=300, runner=None
):
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    server = runner or AppServer(model, effort, log_dir)
    lock = asyncio.Lock()
    app = web.Application(client_max_size=32 * 1024**2)

    async def health(request):
        return web.json_response(
            {
                "status": "ok",
                "model": model,
                "transport": "codex-app-server",
                "busy": lock.locked(),
            }
        )

    async def models(request):
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "owned_by": "codex"}],
            }
        )

    async def complete(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
            convert_messages(payload.get("messages"))
            if payload.get("stream"):
                raise ValueError("streaming is unsupported")
            if payload.get("model", model) not in {model, "codex"}:
                raise ValueError("requested model differs from configured Codex model")
        except (ValueError, TypeError) as error:
            return web.json_response({"error": {"message": str(error)}}, status=400)
        request_id = "chatcmpl-" + uuid.uuid4().hex
        request_dir = log_dir / request_id
        request_dir.mkdir()
        (request_dir / "request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        started = time.monotonic()
        async with lock:
            try:
                result = await asyncio.to_thread(
                    server.run, payload, request_dir, timeout
                )
            except asyncio.CancelledError:
                server.close()
                raise
            except Exception as error:
                record = {"error": str(error), "latency_s": time.monotonic() - started}
                (request_dir / "result.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
                return web.json_response({"error": {"message": str(error)}}, status=503)
        record = {
            **result,
            "latency_s": time.monotonic() - started,
            "effort": effort,
            "requested_max_tokens": payload.get(
                "max_tokens", payload.get("max_completion_tokens")
            ),
            "token_limit_applied": False,
            "requested_temperature": payload.get("temperature"),
            "temperature_applied": False,
        }
        (request_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "codex": {
                key: record[key]
                for key in ("thread_id", "turn_id", "latency_s", "token_limit_applied")
            },
        }
        return web.json_response(response)

    async def cleanup(application):
        await asyncio.to_thread(server.close)

    app.router.add_get("/healthz", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", complete)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    web.run_app(
        build_app(
            model=args.model,
            effort=args.effort,
            log_dir=args.log_dir,
            timeout=args.timeout,
        ),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
