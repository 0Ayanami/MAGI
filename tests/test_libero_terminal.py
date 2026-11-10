"""CPU regressions for opt-in post-success LIBERO observation audits."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vlm_orchestrator.aux_benchmarks.libero_terminal import (
    capture_terminal_observations,
)


class Environment:
    def __init__(self, *, success=None):
        self.actions = []
        self.success = iter(success or [True] * 100)

    def step(self, action):
        self.actions.append(list(action))
        return {"frame": len(self.actions)}, 0, next(self.success), {}


class Client:
    def __init__(self, *, fail_at=None):
        self.queries = []
        self.fail_at = fail_at

    def infer(self, observation):
        self.queries.append(deepcopy(observation))
        if self.fail_at == len(self.queries):
            raise OSError("transport disconnected")
        return {
            "actions": np.full((5, 7), 99.0),
            "orchestrator_instruction": "place the bowl",
            "orchestrator_memory_hold": False,
        }


def builder(gripper=0.0):
    def build(observation):
        return {
            "observation/gripper_position": np.array([gripper]),
            "observation/image": observation["frame"],
            "__episode_id": "2_0",
            "ground_truth_done": True,
        }

    return build


def capture(env, client, *, count=2, interval=3, build=None, record=None):
    return capture_terminal_observations(
        env,
        client,
        {"frame": "actual-success"},
        start_step=11,
        count=count,
        step_interval=interval,
        build_observation=build or builder(),
        record_frame=record,
    )


def test_default_zero_audit_does_not_touch_environment_client_or_builder():
    def unexpected(*args):
        raise AssertionError("disabled audit invoked a callback")

    assert capture(None, None, count=0, build=unexpected, record=unexpected) == {}


@pytest.mark.parametrize(("gripper", "command"), [(0.0, -1.0), (1.0, 1.0)])
def test_fresh_success_and_hold_frames_discard_returned_policy_actions(
    gripper, command
):
    env, client = Environment(), Client()
    frames = []
    result = capture(
        env,
        client,
        build=builder(gripper),
        record=lambda wire, step, response: frames.append(
            (wire["observation/image"], step)
        ),
    )
    assert result["complete"]
    assert [row["step"] for row in result["observations"]] == [11, 14]
    assert [query["__step"] for query in client.queries] == [11, 14]
    assert [query["__episode_id"] for query in client.queries] == ["2_0", "2_0"]
    assert client.queries[0]["observation/image"] == "actual-success"
    assert client.queries[1]["observation/image"] == 3
    assert all(query["__terminal_observation"] is True for query in client.queries)
    assert all("ground_truth_done" not in query for query in client.queries)
    assert env.actions == [[0.0] * 6 + [command]] * 3
    assert result["held_steps"] == 3
    assert result["end_step"] == 14
    assert all(row["policy_actions_executed"] == 0 for row in result["observations"])
    assert all("actions" not in row["orchestrator"] for row in result["observations"])
    assert frames == [("actual-success", 11), (1, 12), (2, 13), (3, 14)]


def test_one_terminal_sample_does_not_step_the_simulation():
    env, client = Environment(), Client()
    result = capture(env, client, count=1)
    assert result["complete"]
    assert len(client.queries) == 1
    assert env.actions == []
    assert result["held_steps"] == 0
    assert result["end_step"] == 11


def test_post_success_instability_is_recorded_separately():
    env, client = Environment(success=[True, False, True]), Client()
    result = capture(env, client)
    assert result["complete"]
    assert result["stability_success"] is True
    assert result["all_held_steps_success"] is False
    assert result["observations"][0]["environment_success"] is True
    assert result["observations"][1]["environment_success"] is True


def test_transport_failure_stops_audit_without_any_returned_action_execution():
    env, client = Environment(), Client(fail_at=2)
    result = capture(env, client, count=3)
    assert result["complete"] is False
    assert result["held_steps"] == 3
    assert len(result["observations"]) == 1
    assert result["error"] == "OSError: transport disconnected"
    assert env.actions == [[0.0] * 6 + [-1.0]] * 3


@pytest.mark.parametrize("gripper", [None, [], [0, 1], float("nan"), 1.5])
def test_invalid_gripper_data_never_produces_a_hold_or_query(gripper):
    env, client = Environment(), Client()
    result = capture(
        env,
        client,
        build=lambda obs: {"observation/gripper_position": gripper},
    )
    assert result["complete"] is False
    assert "gripper" in result["error"]
    assert env.actions == []
    assert client.queries == []


@pytest.mark.parametrize(
    ("count", "interval"),
    [(-1, 1), (True, 1), (2, 0), (2, 1.5)],
)
def test_invalid_audit_limits_are_rejected(count, interval):
    with pytest.raises(ValueError):
        capture(None, None, count=count, interval=interval)


@pytest.mark.parametrize("terminal_count", [0, 2])
def test_eval_entry_preserves_primary_success_when_audit_holds_disturb_scene(
    monkeypatch,
    tmp_path,
    terminal_count,
):
    # Dependency doubles allow the real client control loop to run on CPU.
    # All VLA actions and hold actions still traverse env.step below.
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "imageio", ModuleType("imageio"))
    name = "_terminal_client_entry_test"
    path = Path(__file__).resolve().parents[1] / "examples/libero/libero_eval_client.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    env = Environment(success=[True, False, False])
    env.reset = lambda: {"frame": 0}
    env.set_init_state = lambda state: {"frame": 0}
    suite = SimpleNamespace(
        n_tasks=1,
        get_task=lambda index: SimpleNamespace(
            name="bowl", language="place bowl on plate"
        ),
        get_task_init_states=lambda index: [None],
    )
    benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {"libero_spatial": lambda: suite}
    )
    libero = ModuleType("libero")
    libero_inner = ModuleType("libero.libero")
    libero.libero = libero_inner
    libero_inner.benchmark = benchmark
    monkeypatch.setitem(sys.modules, "libero", libero)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_inner)

    client = Client()
    original_infer = client.infer

    def infer(wire):
        return {
            **original_infer(wire),
            "orchestrator_episode_log_dir": str(tmp_path),
        }

    client.infer = infer
    openpi_client = ModuleType("openpi_client")
    openpi_client.websocket_client_policy = SimpleNamespace(
        WebsocketClientPolicy=lambda *args: client,
    )
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)
    from vlm_orchestrator.aux_benchmarks import robocerebra

    monkeypatch.setattr(robocerebra, "register", lambda: None)
    monkeypatch.setattr(
        module,
        "get_libero_env",
        lambda *args, **kwargs: (env, "place bowl on plate", None, None, 0.01, 10),
    )

    def wire(observation, *args, **kwargs):
        return {
            **builder()(observation),
            "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        }

    monkeypatch.setattr(module, "build_wire_obs", wire)
    from vlm_orchestrator.utils import video_annotator

    class Writer:
        def __init__(self, *args, **kwargs):
            self.frames = []

        def add_frame(self, frame, **kwargs):
            self.frames.append(frame)

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(video_annotator, "RawVideoWriter", Writer)
    monkeypatch.setattr(video_annotator, "EpisodeVideoWriter", Writer)
    args = module.Args(
        task_suite_name="libero_spatial",
        num_steps_wait=0,
        replan_steps=1,
        num_trials_per_task=1,
        terminal_observations=terminal_count,
        terminal_observation_interval=2,
    )
    module.eval_libero(args)
    metadata = json.loads((tmp_path / "eval_metadata.json").read_text())
    assert metadata["success"] is True
    assert metadata["num_steps"] == 0  # Preserve existing client counting.
    assert env.actions[0] == [99.0] * 7
    if terminal_count == 0:
        assert len(env.actions) == len(client.queries) == 1
        assert "terminal_verification" not in metadata
    else:
        assert len(client.queries) == 3
        assert env.actions[1:] == [[0.0] * 6 + [-1.0]] * 2
        assert metadata["terminal_steps"] == 2
        assert metadata["terminal_stability_success"] is False
        audit = metadata["terminal_verification"]
        assert audit["complete"]
        assert [row["step"] for row in audit["observations"]] == [1, 3]
