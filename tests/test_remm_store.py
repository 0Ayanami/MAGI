"""CPU checks of revisable evidence and support semantics."""

import json
import unittest
from dataclasses import FrozenInstanceError

from vlm_orchestrator.remm_memory import (
    AssertionStatus,
    Evidence,
    MemoryStore,
    Predicate,
    Truth,
)


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryStore("episode-1")
        self.inside = Predicate("in", ("red", "bowl"))
        self.at_region = Predicate("at", ("bowl", "back"))

    def evidence(self, step, *, source="geometry", objects=("red", "bowl"), suffix=""):
        event = Evidence(f"frame-{step}{suffix}", step, source, objects, f"rgb-{step}")
        return self.memory.add_evidence(event)

    def assert_direct(self, predicate, truth, step, **kwargs):
        event = self.evidence(step, suffix=str(len(self.memory.events)))
        return self.memory.assert_fact(predicate, truth, (event.id,), **kwargs)

    def test_evidence_copies_and_freezes_nested_payload(self):
        payload = {"position": [0.0, 0.1, 0.2], "details": {"visible": True}}
        event = Evidence("image-1", 1, "geometry", ("red",), payload=payload)
        self.memory.add_evidence(event)
        payload["position"][0] = 10
        payload["details"]["visible"] = False
        self.assertEqual(event.payload["position"], (0.0, 0.1, 0.2))
        self.assertTrue(event.payload["details"]["visible"])
        with self.assertRaises(TypeError):
            event.payload["details"]["visible"] = False
        with self.assertRaises(FrozenInstanceError):
            event.step = 2

    def test_evidence_retry_is_idempotent_and_conflict_is_atomic(self):
        event = self.evidence(1)
        self.assertIs(self.memory.add_evidence(event), event)
        before = self.memory.snapshot()
        with self.assertRaises(ValueError):
            self.memory.add_evidence(Evidence(event.id, 2, "geometry"))
        self.assertEqual(self.memory.snapshot(), before)
        self.assertEqual(len(self.memory.events), 1)

    def test_revision_invalidates_transitive_conjunctive_support_only(self):
        inside = self.assert_direct(self.inside, Truth.TRUE, 1, confidence=0.9)
        region = self.assert_direct(self.at_region, Truth.TRUE, 1, confidence=0.8)
        red_location = Predicate("at", ("red", "back"))
        derived = self.memory.derive(red_location, (inside, region))
        completion = self.memory.complete_goal("transport", (red_location,))
        other = Predicate("on", ("blue", "shelf"))
        self.assert_direct(other, Truth.TRUE, 1)
        self.memory.complete_goal("unrelated", (other,))

        negative = self.assert_direct(self.inside, Truth.FALSE, 2)

        self.assertEqual(negative.truth, Truth.FALSE)
        self.assertEqual(
            self.memory.assertions[inside.id].status, AssertionStatus.SUPERSEDED
        )
        self.assertEqual(
            self.memory.assertions[derived.id].status, AssertionStatus.INVALIDATED
        )
        self.assertEqual(
            self.memory.assertions[completion.id].status, AssertionStatus.INVALIDATED
        )
        self.assertEqual(self.memory.current_truth(red_location), Truth.UNKNOWN)
        self.assertEqual(self.memory.broken_goals, ("transport",))
        self.assertEqual(self.memory.current(self.at_region).id, region.id)
        self.assertEqual(
            self.memory.current_truth(Predicate("satisfied", ("unrelated",))),
            Truth.TRUE,
        )
        self.assertEqual(derived.confidence, 0.8)
        self.assertEqual(derived.parents, (inside.id, region.id))

    def test_corroboration_preserves_completion_support(self):
        first = self.assert_direct(self.inside, Truth.TRUE, 1)
        completion = self.memory.complete_goal("place", (self.inside,))
        repeated = self.assert_direct(self.inside, Truth.TRUE, 2)
        self.assertEqual(first.id, repeated.id)
        self.assertEqual(
            self.memory.assertions[completion.id].status, AssertionStatus.ACTIVE
        )
        self.assertFalse(self.memory.broken_goals)
        self.assertEqual(len(self.memory.events), 2)

    def test_completion_records_latest_verification_boundary_and_evidence(self):
        original = self.assert_direct(self.inside, Truth.TRUE, 1)
        self.assert_direct(self.inside, Truth.TRUE, 2)
        verified = self.evidence(2, source="verification", suffix="completion")
        completion = self.memory.complete_goal(
            "place", (self.inside,), evidence_ids=(verified.id,)
        )
        self.assertEqual(original.step, 1)
        self.assertEqual(completion.step, 2)
        self.assertEqual(completion.evidence_ids, (verified.id,))
        self.assertEqual(completion.parents, (original.id,))

    def test_old_fact_retry_never_resurrects_a_superseded_assertion(self):
        event = self.evidence(1)
        first = self.memory.assert_fact(self.inside, Truth.TRUE, (event.id,))
        self.assert_direct(self.inside, Truth.FALSE, 2)
        retried = self.memory.assert_fact(self.inside, Truth.TRUE, (event.id,))
        self.assertEqual(retried.id, first.id)
        self.assertEqual(retried.status, AssertionStatus.SUPERSEDED)
        self.assertEqual(self.memory.current_truth(self.inside), Truth.FALSE)

    def test_stale_fact_proposal_rejected_after_corroboration(self):
        self.assert_direct(self.inside, Truth.TRUE, 1)
        self.assert_direct(self.inside, Truth.TRUE, 3)
        event = self.evidence(2)
        before = self.memory.snapshot()
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.memory.assert_fact(self.inside, Truth.FALSE, (event.id,))
        self.assertEqual(before, self.memory.snapshot())

    def test_repair_requires_new_evidence_and_can_complete_again(self):
        first = self.assert_direct(self.inside, Truth.TRUE, 1)
        self.memory.complete_goal("place", (self.inside,))
        self.assert_direct(self.inside, Truth.FALSE, 2)
        restored = self.assert_direct(self.inside, Truth.TRUE, 3)
        self.assertNotEqual(first.id, restored.id)
        self.memory.complete_goal("place", (self.inside,))
        self.assertFalse(self.memory.broken_goals)
        self.assertEqual(len(self.memory.assertions), 5)

    def test_unknown_withdraws_completion_without_inventing_false(self):
        self.assert_direct(self.inside, Truth.TRUE, 1)
        self.memory.complete_goal("place", (self.inside,))
        self.assert_direct(self.inside, Truth.UNKNOWN, 2)
        self.assertEqual(self.memory.current_truth(self.inside), Truth.UNKNOWN)
        self.assertEqual(
            self.memory.current_truth(Predicate("satisfied", ("place",))), Truth.UNKNOWN
        )
        self.assertEqual(self.memory.broken_goals, ("place",))

    def test_occlusion_preserves_initial_pose_last_trusted_pose_and_fact(self):
        event = self.evidence(1, objects=("red",))
        initial = self.memory.observe_object(
            "red",
            step=1,
            evidence_ids=(event.id,),
            label="block",
            aliases=("red block",),
            position=(0, 0, 0.1),
        )
        self.memory.assert_fact(self.inside, Truth.TRUE, (event.id,))
        self.memory.complete_goal("place", (self.inside,))
        moved = self.evidence(2, objects=("red",))
        self.memory.observe_object(
            "red", step=2, evidence_ids=(moved.id,), position=(0.1, 0.2, 0.1)
        )
        missing = self.evidence(3, objects=("red",))
        hidden = self.memory.observe_object(
            "red", step=3, evidence_ids=(missing.id,), visible=False
        )
        self.assertFalse(hidden.visible)
        self.assertEqual(hidden.initial_position, initial.initial_position)
        self.assertEqual(hidden.last_trusted_position, (0.1, 0.2, 0.1))
        self.assertEqual(hidden.trusted_step, 2)
        self.assertEqual(self.memory.current_truth(self.inside), Truth.TRUE)
        self.assertFalse(self.memory.broken_goals)
        self.assertEqual(self.memory.resolve_object("red block"), hidden)

    def test_uncertain_identity_cannot_update_trusted_pose(self):
        event = self.evidence(1, objects=("red",))
        self.memory.observe_object(
            "red", step=1, evidence_ids=(event.id,), position=(0, 0, 0)
        )
        uncertain = self.evidence(2, objects=("red",))
        record = self.memory.observe_object(
            "red",
            step=2,
            evidence_ids=(uncertain.id,),
            position=(10, 10, 10),
            identity_confident=False,
        )
        self.assertEqual(record.last_trusted_position, (0.0, 0.0, 0.0))
        self.assertEqual(record.trusted_step, 1)
        self.assertFalse(record.identity_confident)

    def test_object_retry_is_idempotent_and_stale_change_is_rejected(self):
        old = self.evidence(1, objects=("red",))
        first = self.memory.observe_object(
            "red", step=1, evidence_ids=(old.id,), position=(0, 0, 0)
        )
        new = self.evidence(2, objects=("red",))
        latest = self.memory.observe_object(
            "red", step=2, evidence_ids=(new.id,), position=(1, 1, 1)
        )
        retried = self.memory.observe_object(
            "red", step=1, evidence_ids=(old.id,), position=(0, 0, 0)
        )
        self.assertEqual(retried, latest)
        self.assertNotEqual(first, latest)
        with self.assertRaises(ValueError):
            self.memory.observe_object(
                "red", step=1, evidence_ids=(old.id,), position=(5, 5, 5)
            )
        stale = self.evidence(0, objects=("red",))
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.memory.observe_object(
                "red", step=0, evidence_ids=(stale.id,), position=(5, 5, 5)
            )
        self.assertEqual(self.memory.objects["red"], latest)

    def test_alias_collisions_do_not_change_existing_identity(self):
        event = self.evidence(1, objects=("red", "blue"))
        self.memory.observe_object(
            "red", step=1, evidence_ids=(event.id,), aliases=("block",)
        )
        before = self.memory.snapshot()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.memory.observe_object(
                "blue", step=1, evidence_ids=(event.id,), aliases=("block",)
            )
        self.assertEqual(self.memory.snapshot(), before)

    def test_invalid_values_or_missing_evidence_do_not_mutate_state(self):
        event = self.evidence(1, objects=("red",))
        before = self.memory.snapshot()
        for confidence in (float("nan"), float("inf"), -1, 1.1, True, "high"):
            with self.assertRaises(ValueError):
                self.memory.assert_fact(
                    self.inside, Truth.TRUE, (event.id,), confidence=confidence
                )
        with self.assertRaises(ValueError):
            self.memory.assert_fact(self.inside, Truth.TRUE, ("missing",))
        with self.assertRaises(ValueError):
            self.memory.assert_fact(self.inside, Truth.TRUE)
        with self.assertRaises(ValueError):
            self.memory.observe_object(
                "red", step=1, evidence_ids=(event.id,), position=(0, float("nan"), 0)
            )
        self.assertEqual(self.memory.snapshot(), before)
        with self.assertRaises(ValueError):
            Evidence("bad", 1, "geometry", payload={"position": [float("nan")]})

    def test_goal_requires_current_physical_postconditions(self):
        with self.assertRaises(ValueError):
            self.memory.complete_goal("place", ())
        with self.assertRaises(ValueError):
            self.memory.complete_goal("place", (self.inside,))
        event = self.evidence(1)
        with self.assertRaises(ValueError):
            self.memory.assert_fact(
                Predicate("satisfied", ("place",)), Truth.TRUE, (event.id,)
            )
        self.memory.assert_fact(self.inside, Truth.FALSE, (event.id,))
        with self.assertRaises(ValueError):
            self.memory.complete_goal("place", (self.inside,))

    def test_parents_must_be_supported_and_cannot_depend_on_replaced_fact(self):
        first = self.assert_direct(self.inside, Truth.TRUE, 1)
        conclusion = self.memory.derive(self.at_region, (first,))
        before = self.memory.snapshot()
        with self.assertRaises(ValueError):
            self.memory.derive(self.inside, (conclusion,))
        self.assertEqual(self.memory.snapshot(), before)
        self.assert_direct(self.inside, Truth.FALSE, 2)
        with self.assertRaises(ValueError):
            self.memory.derive(Predicate("at", ("red", "back")), (first,))

    def test_snapshot_is_json_safe_and_cannot_mutate_the_store(self):
        self.assert_direct(self.inside, Truth.TRUE, 1)
        self.memory.complete_goal("place", (self.inside,))
        snapshot = self.memory.snapshot()
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        snapshot["events"].clear()
        snapshot["current_facts"][0]["evidence_ids"].clear()
        self.assertEqual(len(self.memory.events), 1)
        self.assertTrue(self.memory.current(self.inside).evidence_ids)
        current = self.memory.context(history=False)
        self.assertNotIn("events", current)
        self.assertNotIn("assertions", current)

    def test_episode_stores_do_not_share_history(self):
        self.assert_direct(self.inside, Truth.TRUE, 1)
        fresh = MemoryStore("episode-2")
        self.assertFalse(fresh.events)
        self.assertFalse(fresh.objects)
        self.assertEqual(fresh.current_truth(self.inside), Truth.UNKNOWN)
        with self.assertRaises(ValueError):
            fresh.derive(self.at_region, (self.memory.current(self.inside),))

    def test_staging_branch_preserves_live_state_when_a_batch_is_rejected(self):
        original = self.assert_direct(self.inside, Truth.TRUE, 1)
        completion = self.memory.complete_goal("place", (self.inside,))
        before = self.memory.snapshot()
        branch = self.memory.fork()
        self.assertIs(
            branch.assertions[original.id], self.memory.assertions[original.id]
        )
        changed = branch.add_evidence(Evidence("changed", 2, "geometry", ("red",)))
        branch.assert_fact(self.inside, Truth.FALSE, (changed.id,))
        self.assertEqual(branch.broken_goals, ("place",))
        with self.assertRaises(ValueError):
            branch.observe_object("blue", step=2, evidence_ids=(changed.id,))
        self.assertEqual(self.memory.snapshot(), before)
        self.assertEqual(
            self.memory.assertions[completion.id].status, AssertionStatus.ACTIVE
        )

    def test_staging_branches_have_independent_aliases_and_dependency_edges(self):
        original = self.assert_direct(self.inside, Truth.TRUE, 1)
        branch = self.memory.fork()
        goal = branch.complete_goal("branch-only", (self.inside,))
        self.assertNotIn(goal.id, self.memory.assertions)
        event = branch.add_evidence(Evidence("object", 2, "geometry", ("red",)))
        branch.observe_object(
            "red", step=2, evidence_ids=(event.id,), aliases=("red block",)
        )
        self.assertIsNone(self.memory.resolve_object("red block"))
        self.assert_direct(self.inside, Truth.FALSE, 3)
        self.assertEqual(branch.assertions[original.id].status, AssertionStatus.ACTIVE)
        self.assertEqual(branch.assertions[goal.id].status, AssertionStatus.ACTIVE)
        self.assertFalse(branch.broken_goals)


if __name__ == "__main__":
    unittest.main()
