"""Exercise the proxy entry point with CPU transports and real REMM decisions."""

import json
import logging

import numpy as np
import pytest

from vlm_orchestrator.protocols.base import (
    Backend,
    BackendConnection,
    Frontend,
    FrontendSession,
)
from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
from vlm_orchestrator.remm_memory.runtime import RemmConfig
from vlm_orchestrator.strategies.base import StrategyContext
from vlm_orchestrator.strategies.remm_subgoal import RemmSubgoalStrategy
from vlm_orchestrator.strategies.subgoal import SubgoalConfig
from vlm_orchestrator.strategies.subgoal_base import SubgoalBaseStrategy
from vlm_orchestrator.utils import codec
from vlm_orchestrator.vlm import PassthroughVLM

PLAN = {
    "subgoals": ["pick up red block", "place red block on blue block"],
    "ordered": True,
    "memory": {
        "objects": [{"id": "red", "description": "red block"}],
        "goals": [
            {"subgoal_index": 0, "predicates": [{"name": "holding", "args": ["red"]}]}
        ],
    },
}
NEXT = {"status": "complete", "action": "next", "reason": "appears complete"}


def wire_copy(value):
    """Use the real wire representation on both sides of the CPU transport."""
    return codec.unpackb(codec.Packer().pack(value))


class MemorySession(FrontendSession):
    def __init__(self, observations):
        self.observations = iter(observations)
        self.metadata = None
        self.responses = []

    async def recv_obs(self):
        observation = next(self.observations, None)
        return wire_copy(observation) if observation is not None else None

    async def send_metadata(self, metadata):
        self.metadata = wire_copy(metadata)

    async def send_action(self, canonical_action):
        self.responses.append(wire_copy(canonical_action))


class MemoryFrontend(Frontend):
    def __init__(self, session):
        self.session = session
        self.address = None

    async def serve(self, host, port, on_session):
        self.address = (host, port)
        await on_session(self.session)


class RecordingConnection(BackendConnection):
    def __init__(self):
        self.observations = []
        self.closed = False

    async def recv_metadata(self):
        return {"model": "CPU-test-policy", "version": "test"}

    async def infer(self, canonical_obs):
        self.observations.append(wire_copy(canonical_obs))
        return {"actions": np.full((5, 7), 0.25, dtype=np.float32)}

    async def close(self):
        self.closed = True


class RecordingBackend(Backend):
    def __init__(self):
        self.connection = RecordingConnection()

    async def connect(self):
        return self.connection


def observation(step, **extra):
    return {
        "prompt": "move red block onto blue block",
        "__step": step,
        "__episode_id": 1,
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/remm_objects": {
            "coordinate_frame": "world",
            "units": "m",
            "source": "visual",
            "objects": [
                {
                    "id": "red",
                    "position": [0, 0, 0.1],
                    "holding": None,
                    "frame_id": f"frame-{step}",
                }
            ],
        },
        **extra,
    }


def proxy_fixture(monkeypatch, observations, *, gate="enforce", replies=None):
    responses = iter(replies or [PLAN, NEXT, {"done": True}])
    prompts = []

    def model_call(self, system_prompt, content, max_tokens=None):
        prompts.append((system_prompt, content))
        return json.dumps(next(responses))

    monkeypatch.setattr(SubgoalBaseStrategy, "_vlm_call", model_call)
    strategy = RemmSubgoalStrategy(
        StrategyContext(vlm=PassthroughVLM(), image_key="observation/image"),
        SubgoalConfig(check_interval=1),
        memory_config=RemmConfig(gate=gate, max_reobservations=0),
        failure_monitor="vlm",
        recovery_mode="replan",
    )
    session = MemorySession(observations)
    frontend, backend = MemoryFrontend(session), RecordingBackend()
    proxy = OrchestratorProxy(
        ProxyConfig(
            strategy=strategy,
            image_key="observation/image",
            frontend=frontend,
            backend=backend,
            host="127.0.0.1",
            port=8001,
        )
    )
    return proxy, session, frontend, backend.connection, prompts


def test_remm_payload_is_consumed_and_stripped_before_backend_inference(monkeypatch):
    obs = observation(
        0,
        **{
            "observation/remm_debug": "private memory data",
            "__terminal_observation": True,
            "gt_state": {"objects": {"red": {"holding": True}}},
        },
    )
    proxy, session, frontend, backend, _ = proxy_fixture(
        monkeypatch, [obs], gate="shadow"
    )
    proxy.serve_forever()
    assert frontend.address == ("127.0.0.1", 8001)
    assert session.metadata["orchestrator"] is True
    assert session.metadata["model"] == "CPU-test-policy"
    assert len(backend.observations) == len(session.responses) == 1
    forwarded = backend.observations[0]
    assert not any(key.startswith("observation/remm_") for key in forwarded)
    assert "gt_state" not in forwarded
    assert "__terminal_observation" not in forwarded
    assert forwarded["prompt"] == PLAN["subgoals"][0]
    assert np.array_equal(forwarded["observation/image"], obs["observation/image"])
    assert np.all(session.responses[0]["actions"] == 0.25)
    assert backend.closed


@pytest.mark.parametrize(("gripper", "expected_command"), [(0.0, -1.0), (1.0, 1.0)])
def test_failed_memory_repair_holds_without_policy_inference_and_preserves_gripper(
    monkeypatch,
    gripper,
    expected_command,
):
    measured = {"observation/gripper_position": np.array([gripper])}
    observations = [
        observation(0, **measured),
        observation(1, **measured),
        observation(2),
    ]
    proxy, session, _, backend, prompts = proxy_fixture(monkeypatch, observations)
    proxy.serve_forever()
    assert len(prompts) == 3  # Initial plan, monitor, failed repair.
    assert len(backend.observations) == 1
    assert len(session.responses) == 3
    for response in session.responses[1:]:
        assert response["orchestrator_memory_hold"] is True
        assert response["orchestrator_instruction"] == PLAN["subgoals"][0]
        assert response["orchestrator_subgoal_idx"] == 0
        actions = response["actions"]
        assert actions.shape == (5, 7)
        assert np.isfinite(actions).all()
        assert np.all(actions[:, :6] == 0)
        assert np.all(actions[:, 6] == expected_command)
    assert session.responses[1]["orchestrator_flush_actions"] is True
    assert backend.closed


def test_hold_without_any_valid_gripper_signal_reports_error_and_sends_no_guess(
    monkeypatch,
    caplog,
):
    proxy, session, _, backend, _ = proxy_fixture(
        monkeypatch, [observation(0), observation(1)]
    )
    with caplog.at_level(logging.ERROR):
        proxy.serve_forever()
    assert len(session.responses) == 1
    assert len(backend.observations) == 1
    assert backend.closed
    assert any("gripper" in record.message.lower() for record in caplog.records)
