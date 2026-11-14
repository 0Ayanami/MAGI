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
from urllib.parse import urlsplit

from aiohttp import web

from vlm_orchestrator.vlm import parse_json


def _provider_key(env_name: str, env_file: Path) -> str:
    """Read one credential without importing other settings into the process."""
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == env_name:
                key = value.strip()
                if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
                    key = key[1:-1].strip()
                if key:
                    return key
    raise ValueError(f"Set {env_name} in {env_file} or the local environment")


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
        self, model: str, effort: str, log_dir: Path, executable: str = "codex",
        *, responses_base_url: str | None = None,
        provider_key_env: str = "DASHSCOPE_API_KEY",
        model_catalog: Path | None = None,
        env_file: Path = Path(".env"),
    ):
        self.model = model
        self.effort = effort
        self.log_dir = log_dir
        self.executable = executable
        self.responses_base_url = responses_base_url
        self.provider_key_env = provider_key_env
        self.env_file = env_file.resolve()
        self.model_catalog = model_catalog.resolve() if model_catalog else None
        self.process = None
        self.messages = Queue()
        self.sequence = 0
        self.thread_id = None
        self.turn_id = None
        self.stderr = None

    def _startup_args(self, executable):
        args = [executable, "-a", "never"]
        if self.responses_base_url is None:
            args.extend(["-c", "model_providers.openai.supports_websockets=false"])
        else:
            provider = "model_providers.voloagent_api"
            overrides = {
                "model": self.model,
                "model_provider": "voloagent_api",
                "web_search": "disabled",
                "features.shell_tool": False,
                f"{provider}.name": "VoLoAgent Responses API",
                f"{provider}.base_url": self.responses_base_url,
                f"{provider}.env_key": self.provider_key_env,
                f"{provider}.wire_api": "responses",
                f"{provider}.requires_openai_auth": False,
                f"{provider}.supports_websockets": False,
            }
            if self.model_catalog is not None:
                overrides["model_catalog_json"] = str(self.model_catalog)
            for key, value in overrides.items():
                args.extend(["-c", f"{key}={json.dumps(value)}"])
        return [*args, "app-server"]

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
        child_env = None
        if self.responses_base_url is not None:
            child_env = os.environ.copy()
            child_env[self.provider_key_env] = _provider_key(self.provider_key_env, self.env_file)
        self.messages = Queue()
        self.stderr = (self.log_dir / "app-server.stderr.log").open(
            "a", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            self._startup_args(executable),
            env=child_env,
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

    def _quota_snapshot(self, phase, request_dir, deadline, audit):
        """Record account counters without guessing per-turn credit charges."""
        started = time.monotonic()
        snapshot = {"timestamp": time.time(), "scope": "account"}
        if self.responses_base_url is not None:
            snapshot.update(status="not_applicable", scope="external_provider")
        else:
            try:
                snapshot["rate_limits"] = self._request(
                    "account/rateLimits/read", {},
                    min(deadline, time.monotonic() + 5), audit,
                )
                snapshot["status"] = "available"
            except (RuntimeError, TimeoutError) as error:
                snapshot.update(status="unavailable", error=str(error))
        snapshot["latency_s"] = time.monotonic() - started
        (request_dir / f"quota-{phase}.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return snapshot

    def run(self, payload, request_dir, timeout):
        deadline = time.monotonic() + timeout
        inputs = convert_messages(payload["messages"])
        self.thread_id = self.turn_id = None
        with (request_dir / "app-server-events.jsonl").open(
            "w", encoding="utf-8"
        ) as audit:
            try:
                self._start(deadline, audit)
                quota_before = self._quota_snapshot("before", request_dir, deadline, audit)
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
                provider = started.get("modelProvider") or started["thread"].get("modelProvider")
                if self.responses_base_url is not None and (
                    provider != "voloagent_api" or model != self.model
                ):
                    raise RuntimeError("Codex selected a different model or API provider")
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
                turn_started = time.monotonic()
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
                turn_latency = time.monotonic() - turn_started
                quota_after = self._quota_snapshot("after", request_dir, deadline, audit)
                return {
                    "text": json.dumps(parsed, ensure_ascii=False),
                    "raw_text": final,
                    "thread_id": self.thread_id,
                    "turn_id": self.turn_id,
                    "model": model,
                    "model_provider": provider,
                    "usage": usage,
                    "turn_latency_s": turn_latency,
                    "quota_before": quota_before,
                    "quota_after": quota_after,
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
    *, model="gpt-6.1-sol", effort="low", log_dir: Path, timeout=300, runner=None,
    responses_base_url: str | None = None, provider_key_env="DASHSCOPE_API_KEY",
    model_catalog: Path | None = None,
    env_file: Path = Path(".env"),
):
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    server = runner or AppServer(
        model, effort, log_dir, responses_base_url=responses_base_url,
        provider_key_env=provider_key_env, model_catalog=model_catalog, env_file=env_file,
    )
    lock = asyncio.Lock()
    app = web.Application(client_max_size=32 * 1024**2)

    async def health(request):
        return web.json_response(
            {
                "status": "ok",
                "model": model,
                "transport": "codex-app-server",
                "inference_backend": "responses_api" if responses_base_url else "codex_account",
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
            acquired = time.monotonic()
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
            "queue_wait_s": acquired - started,
            "service_latency_s": time.monotonic() - acquired,
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
    parser.add_argument("--responses-base-url", help="Custom provider base URL, excluding /responses")
    parser.add_argument("--provider-key-env", default="DASHSCOPE_API_KEY",
                        help="Name of the provider API key setting")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="Local credential file (default: .env in the launch directory)")
    parser.add_argument("--model-catalog", type=Path,
                        help="Optional Codex model catalog JSON for the custom provider")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if args.responses_base_url:
        url = urlsplit(args.responses_base_url)
        loopback = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            not url.hostname or (url.scheme != "https" and not (url.scheme == "http" and loopback))
            or url.username is not None or url.password is not None or url.query or url.fragment
            or url.path.rstrip("/").endswith(("/responses", "/chat/completions"))
        ):
            parser.error("use the provider's HTTPS base URL without credentials or an API operation suffix")
        try:
            _provider_key(args.provider_key_env, args.env_file)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        args.responses_base_url = args.responses_base_url.rstrip("/")
    if args.model_catalog and (not args.responses_base_url or not args.model_catalog.is_file()):
        parser.error("--model-catalog requires a custom Responses base URL and an existing JSON file")
    web.run_app(
        build_app(
            model=args.model,
            effort=args.effort,
            log_dir=args.log_dir,
            timeout=args.timeout,
            responses_base_url=args.responses_base_url,
            provider_key_env=args.provider_key_env,
            model_catalog=args.model_catalog,
            env_file=args.env_file,
        ),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
