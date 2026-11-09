"""CPU protocol and HTTP regressions for the Codex decision bridge."""

import asyncio
import json
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from vlm_orchestrator.utils.codex_vlm_server import (
    AppServer,
    build_app,
    convert_messages,
)


def test_message_conversion_preserves_roles_text_and_all_images():
    images = ["data:image/png;base64,YQ==", "data:image/jpeg;base64,Yg=="]
    converted = convert_messages(
        [
            {"role": "system", "content": "return action JSON"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": images[0]}},
                    {"type": "text", "text": "now"},
                    {"type": "image_url", "image_url": {"url": images[1]}},
                ],
            },
        ]
    )
    assert [part["url"] for part in converted if part["type"] == "image"] == images
    text = " ".join(part["text"] for part in converted if part["type"] == "text")
    assert "[system message] return action JSON [user message] before now" in text


@pytest.mark.parametrize(
    "messages",
    [
        [],
        None,
        [{"role": "tool", "content": "x"}],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://unrelated.example/image"},
                    }
                ],
            }
        ],
    ],
)
def test_invalid_input_cannot_be_silently_dropped(messages):
    with pytest.raises(ValueError):
        convert_messages(messages)


class ScriptedAppServer(AppServer):
    def __init__(self, messages, directory):
        super().__init__("gpt-6.1-sol", "low", directory)
        self.notifications = iter(messages)
        self.closed = False
        self.requests = []

    def _start(self, deadline, audit):
        pass

    def _request(self, method, params, deadline, audit):
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread1"}, "model": self.model}
        return {"turn": {"id": "turn1"}}

    def _receive(self, deadline, audit):
        return next(self.notifications)

    def close(self):
        self.closed = True


def test_app_server_turn_records_actual_result_and_usage(tmp_path):
    messages = [
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread1",
                "tokenUsage": {"total": {"inputTokens": 20}},
            },
        },
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread1",
                "item": {"type": "agentMessage", "text": '{"action":"next"}'},
            },
        },
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread1",
                "turn": {"id": "turn1", "status": "completed"},
            },
        },
    ]
    server = ScriptedAppServer(messages, tmp_path)
    result = server.run(
        {"messages": [{"role": "user", "content": "verify"}]}, tmp_path, 10
    )
    assert json.loads(result["text"]) == {"action": "next"}
    assert result["usage"]["total"]["inputTokens"] == 20
    assert result["thread_id"] == "thread1"
    assert server.requests[0][1]["ephemeral"] is True
    assert server.requests[1][1]["effort"] == "low"


@pytest.mark.parametrize(
    "item",
    [
        {"type": "agentMessage", "text": "This is not valid JSON"},
        {"type": "commandExecution", "command": "unrequested tool"},
    ],
)
def test_invalid_decision_or_tool_activity_closes_process(tmp_path, item):
    server = ScriptedAppServer(
        [
            {"method": "item/completed", "params": {"item": item}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ],
        tmp_path,
    )
    with pytest.raises((ValueError, RuntimeError)):
        server.run({"messages": [{"role": "user", "content": "verify"}]}, tmp_path, 10)
    assert server.closed


class DecisionRunner:
    def __init__(self, fail=False):
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.closed = False

    def run(self, payload, directory, timeout):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            time.sleep(0.02)
            if self.fail:
                raise TimeoutError("real process deadline exceeded")
            return {
                "text": '{"action":"continue"}',
                "thread_id": "t",
                "turn_id": "u",
                "model": "gpt-6.1-sol",
                "usage": None,
            }
        finally:
            self.active -= 1

    def close(self):
        self.closed = True


def test_http_serialization_and_audit_without_fake_token_truncation(tmp_path):
    async def scenario():
        runner = DecisionRunner()
        client = TestClient(TestServer(build_app(log_dir=tmp_path, runner=runner)))
        await client.start_server()
        try:
            health = await client.get("/healthz")
            assert (await health.json())["transport"] == "codex-app-server"
            payload = {
                "model": "codex",
                "messages": [{"role": "user", "content": "check"}],
                "max_tokens": 512,
                "temperature": 0.0,
            }
            responses = await asyncio.gather(
                *(client.post("/v1/chat/completions", json=payload) for _ in range(2))
            )
            for response in responses:
                assert response.status == 200
                data = await response.json()
                assert data["choices"][0]["finish_reason"] == "stop"
                assert data["codex"]["token_limit_applied"] is False
            assert runner.max_active == 1
            logs = list(tmp_path.glob("chatcmpl-*/result.json"))
            assert len(logs) == 2
            assert json.loads(logs[0].read_text())["requested_max_tokens"] == 512
            record = json.loads(logs[0].read_text())
            assert record["requested_temperature"] == 0.0
            assert record["temperature_applied"] is False
        finally:
            await client.close()
        assert runner.closed

    asyncio.run(scenario())


def test_http_timeout_returns_error_and_audit(tmp_path):
    async def scenario():
        runner = DecisionRunner(fail=True)
        client = TestClient(TestServer(build_app(log_dir=tmp_path, runner=runner)))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "verify"}]},
            )
            assert response.status == 503
            assert "deadline" in (await response.json())["error"]["message"]
            result = json.loads(
                next(tmp_path.glob("chatcmpl-*/result.json")).read_text()
            )
            assert "deadline" in result["error"]
            response = await client.post(
                "/v1/chat/completions", json={"messages": [], "stream": True}
            )
            assert response.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())
