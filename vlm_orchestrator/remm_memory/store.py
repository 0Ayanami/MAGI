"""Evidence ledger and supported-state projection inspired by REMM, section III.

Source-specific perception gates belong to the caller. This store accepts typed
evidence, preserves it, and withdraws derived assertions when support changes.
It has no dependency on a policy, simulator, ground truth, or model service.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from numbers import Real
from types import MappingProxyType

from .types import (
    Assertion,
    AssertionStatus,
    Evidence,
    ObjectRecord,
    Predicate,
    Truth,
    identifier,
    identifiers,
    step_number,
)


class MemoryStore:
    """An episode's accepted evidence and conjunctively supported assertions.

    Assertion identifiers always refer to an existing record from this store.
    Adding parents can therefore never introduce a dependency cycle. Replacing
    a supported value invalidates descendants and preserves all prior records.
    Corroboration keeps its assertion identity, so checking an unchanged fact
    does not falsely revoke goals completed using that fact.
    """

    def __init__(self, episode_id: str | int):
        if isinstance(episode_id, bool) or not isinstance(episode_id, (str, int)):
            raise ValueError("episode_id must be a string or integer")
        identifier(str(episode_id), "episode id")
        self.episode_id = episode_id
        self._events: dict[str, Evidence] = {}
        self._objects: dict[str, ObjectRecord] = {}
        self._aliases: dict[str, str] = {}
        self._object_observations: dict[tuple, tuple] = {}
        self._assertions: dict[str, Assertion] = {}
        self._current: dict[Predicate, str] = {}
        self._children: dict[str, set[str]] = {}
        self._proposals: dict[tuple, str] = {}
        self._latest_fact_steps: dict[Predicate, int] = {}
        self._goal_records: dict[str, str] = {}

    def fork(self) -> MemoryStore:
        """Create an isolated staging branch sharing only immutable records.

        A caller can validate a complete incoming observation on this branch
        and replace its live store only after all object and fact updates have
        succeeded. Rejected batches then leave the live projection untouched.
        """
        branch = MemoryStore(self.episode_id)
        branch._events = self._events.copy()
        branch._objects = self._objects.copy()
        branch._aliases = self._aliases.copy()
        branch._object_observations = self._object_observations.copy()
        branch._assertions = self._assertions.copy()
        branch._current = self._current.copy()
        branch._children = {
            record_id: children.copy() for record_id, children in self._children.items()
        }
        branch._proposals = self._proposals.copy()
        branch._latest_fact_steps = self._latest_fact_steps.copy()
        branch._goal_records = self._goal_records.copy()
        return branch

    @property
    def events(self) -> tuple[Evidence, ...]:
        return tuple(self._events.values())

    @property
    def objects(self) -> Mapping[str, ObjectRecord]:
        return MappingProxyType(self._objects)

    @property
    def assertions(self) -> Mapping[str, Assertion]:
        return MappingProxyType(self._assertions)

    @property
    def broken_goals(self) -> tuple[str, ...]:
        return tuple(
            goal
            for goal, record_id in self._goal_records.items()
            if self._assertions[record_id].status == AssertionStatus.INVALIDATED
        )

    def add_evidence(self, evidence: Evidence) -> Evidence:
        """Append accepted evidence exactly once, rejecting conflicting retries."""
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence record")
        old = self._events.get(evidence.id)
        if old is not None:
            if old != evidence:
                raise ValueError(f"Conflicting evidence id: {evidence.id}")
            return old
        self._events[evidence.id] = evidence
        return evidence

    def _evidence(self, evidence_ids) -> tuple[Evidence, ...]:
        references = identifiers(evidence_ids, "evidence ids")
        missing = set(references) - self._events.keys()
        if missing:
            raise ValueError(f"Unknown evidence ids: {sorted(missing)}")
        return tuple(self._events[reference] for reference in references)

    def observe_object(
        self,
        object_id: str,
        *,
        step: int,
        evidence_ids=(),
        label: str | None = None,
        aliases=(),
        position=None,
        visible: bool = True,
        identity_confident: bool = True,
    ) -> ObjectRecord:
        """Update visibility and retain the last identity-grounded position.

        Missing or uncertain detections do not retract physical relations. The
        caller can use the visibility and trusted observation age to return an
        UNKNOWN completion check without converting occlusion into motion.
        """
        identifier(object_id, "object id")
        step_number(step)
        evidence = self._evidence(evidence_ids)
        if not evidence:
            raise ValueError("Object observations require evidence")
        if any(event.step != step for event in evidence):
            raise ValueError("Object evidence must come from the observation step")
        if any(
            event.object_ids and object_id not in event.object_ids for event in evidence
        ):
            raise ValueError(
                "Object observation references evidence for another object"
            )
        if not isinstance(visible, bool) or not isinstance(identity_confident, bool):
            raise ValueError("visibility and identity confidence must be booleans")
        if label is not None:
            identifier(label, "object label")
        aliases = identifiers(aliases, "object aliases")
        if object_id in aliases:
            aliases = tuple(alias for alias in aliases if alias != object_id)
        for alias in (object_id, *aliases):
            owner = self._aliases.get(alias)
            if owner is not None and owner != object_id:
                raise ValueError(f"Ambiguous object identity: {alias}")
        if position is not None:
            if isinstance(position, (str, bytes)):
                raise ValueError("Position must have three finite coordinates")
            position = tuple(position)
            if len(position) != 3 or not all(
                isinstance(coordinate, Real)
                and not isinstance(coordinate, bool)
                and math.isfinite(coordinate)
                for coordinate in position
            ):
                raise ValueError("Position must have three finite coordinates")
            position = tuple(float(coordinate) for coordinate in position)
        event_ids = tuple(event.id for event in evidence)
        proposal = (label, aliases, position, visible, identity_confident)
        observation_key = (object_id, step, event_ids)
        prior_proposal = self._object_observations.get(observation_key)
        if prior_proposal is not None:
            if prior_proposal != proposal:
                raise ValueError("Conflicting replay of an object observation")
            return self._objects[object_id]
        previous = self._objects.get(object_id)
        if previous is not None and step < previous.observed_step:
            raise ValueError("Stale object observation")

        trusted = visible and identity_confident and position is not None
        if previous is None:
            record = ObjectRecord(
                id=object_id,
                label=label,
                aliases=aliases,
                initial_position=None,
                last_trusted_position=None,
                visible=visible,
                identity_confident=identity_confident,
                observed_step=step,
                trusted_step=None,
                evidence_ids=event_ids,
                trusted_evidence_ids=(),
            )
        else:
            record = replace(
                previous,
                label=label if label is not None else previous.label,
                aliases=tuple(dict.fromkeys((*previous.aliases, *aliases))),
                visible=visible,
                identity_confident=identity_confident,
                observed_step=step,
                evidence_ids=event_ids,
            )
        if trusted:
            record = replace(
                record,
                initial_position=record.initial_position or position,
                last_trusted_position=position,
                trusted_step=step,
                trusted_evidence_ids=event_ids,
            )
        self._objects[object_id] = record
        self._object_observations[observation_key] = proposal
        for alias in (object_id, *record.aliases):
            self._aliases[alias] = object_id
        return record

    def resolve_object(self, reference: str) -> ObjectRecord | None:
        object_id = self._aliases.get(reference)
        return self._objects.get(object_id)

    def current(self, predicate: Predicate) -> Assertion | None:
        record_id = self._current.get(predicate)
        return self._assertions.get(record_id)

    def current_truth(self, predicate: Predicate) -> Truth:
        record = self.current(predicate)
        return record.truth if record is not None else Truth.UNKNOWN

    def _parents(self, parents) -> tuple[Assertion, ...]:
        if isinstance(parents, str):
            raise ValueError("parents must be a sequence of assertion references")
        references = identifiers(
            (
                parent.id if isinstance(parent, Assertion) else parent
                for parent in parents
            ),
            "parent ids",
        )
        records = []
        for reference in references:
            record = self._assertions.get(reference)
            if (
                record is None
                or record.status != AssertionStatus.ACTIVE
                or record.truth != Truth.TRUE
            ):
                raise ValueError(
                    f"Parent must be a currently supported TRUE assertion: {reference}"
                )
            records.append(record)
        return tuple(records)

    def assert_fact(
        self,
        predicate: Predicate,
        truth: Truth,
        evidence_ids=(),
        confidence: float = 1.0,
        parents=(),
    ) -> Assertion:
        """Accept a direct fact or an explicitly justified derived assertion.

        All validation happens before state changes. Exact proposal retries are
        idempotent even after their result was superseded by newer evidence.
        """
        if not isinstance(predicate, Predicate):
            raise TypeError("predicate must be a Predicate")
        truth = Truth(truth)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be finite and between zero and one")
        evidence = self._evidence(evidence_ids)
        if isinstance(parents, str):
            raise ValueError("parents must be a sequence of assertion references")
        parent_ids = tuple(
            parent.id if isinstance(parent, Assertion) else parent for parent in parents
        )
        parent_ids = identifiers(parent_ids, "parent ids")
        if not evidence and not parent_ids:
            raise ValueError("Assertions require evidence or supporting parents")
        if predicate.name == "satisfied" and not parent_ids:
            raise ValueError("Goal completion requires supporting physical assertions")
        event_ids = tuple(event.id for event in evidence)
        proposal = (predicate, truth, event_ids, confidence, parent_ids)
        prior_id = self._proposals.get(proposal)
        if prior_id is not None:
            return self._assertions[prior_id]
        parent_records = self._parents(parent_ids)
        assertion_step = max(
            [event.step for event in evidence]
            + [parent.step for parent in parent_records]
        )
        if assertion_step < self._latest_fact_steps.get(predicate, -1):
            raise ValueError("Stale evidence cannot overwrite a newer fact")
        previous = self.current(predicate)
        if (
            previous is not None
            and previous.truth == truth
            and previous.parents == parent_ids
        ):
            self._proposals[proposal] = previous.id
            self._latest_fact_steps[predicate] = assertion_step
            return previous

        if previous is not None:
            pending = list(parent_ids)
            visited = set()
            while pending:
                parent_id = pending.pop()
                if parent_id == previous.id:
                    raise ValueError(
                        "A fact cannot be replaced using support that depends on itself"
                    )
                if parent_id not in visited:
                    visited.add(parent_id)
                    pending.extend(self._assertions[parent_id].parents)

        record_id = f"{self.episode_id}:assertion:{len(self._assertions) + 1}"
        record = Assertion(
            id=record_id,
            predicate=predicate,
            truth=truth,
            evidence_ids=event_ids,
            confidence=float(confidence),
            parents=parent_ids,
            status=AssertionStatus.ACTIVE,
            step=assertion_step,
        )
        if previous is not None:
            self._assertions[previous.id] = replace(
                previous, status=AssertionStatus.SUPERSEDED
            )
            self._invalidate_descendants(previous.id)
        self._assertions[record_id] = record
        self._current[predicate] = record_id
        self._proposals[proposal] = record_id
        self._latest_fact_steps[predicate] = assertion_step
        for parent in parent_ids:
            self._children.setdefault(parent, set()).add(record_id)
        return record

    def _invalidate_descendants(self, record_id: str) -> None:
        pending = deque(sorted(self._children.get(record_id, ())))
        visited = set()
        while pending:
            child_id = pending.popleft()
            if child_id in visited:
                continue
            visited.add(child_id)
            child = self._assertions[child_id]
            if child.status == AssertionStatus.ACTIVE:
                self._assertions[child_id] = replace(
                    child, status=AssertionStatus.INVALIDATED
                )
                if self._current.get(child.predicate) == child_id:
                    self._current.pop(child.predicate)
            pending.extend(sorted(self._children.get(child_id, ())))

    def derive(self, predicate: Predicate, parents) -> Assertion:
        parent_records = self._parents(parents)
        if not parent_records:
            raise ValueError("Derived assertions require at least one parent")
        return self.assert_fact(
            predicate,
            Truth.TRUE,
            confidence=min(parent.confidence for parent in parent_records),
            parents=parent_records,
        )

    def complete_goal(self, goal_id: str, predicates, evidence_ids=()) -> Assertion:
        identifier(goal_id, "goal id")
        predicates = tuple(predicates)
        if not predicates:
            raise ValueError("Goal completion requires physical postconditions")
        parents = []
        for predicate in predicates:
            record = self.current(predicate)
            if record is None or record.truth != Truth.TRUE:
                raise ValueError(f"Goal postcondition is not supported: {predicate}")
            if record.predicate.name == "satisfied":
                raise ValueError(
                    "Goal postconditions must reference physical assertions"
                )
            parents.append(record)
        completed = self.assert_fact(
            Predicate("satisfied", (goal_id,)),
            Truth.TRUE,
            evidence_ids=evidence_ids,
            confidence=min(parent.confidence for parent in parents),
            parents=parents,
        )
        self._goal_records[goal_id] = completed.id
        return completed

    def snapshot(self, include_history: bool = True) -> dict:
        snapshot = {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "objects": {
                name: record.to_dict() for name, record in self._objects.items()
            },
            "current_facts": [
                self._assertions[record_id].to_dict()
                for record_id in self._current.values()
            ],
            "broken_goals": list(self.broken_goals),
        }
        if include_history:
            snapshot["events"] = [event.to_dict() for event in self._events.values()]
            snapshot["assertions"] = [
                record.to_dict() for record in self._assertions.values()
            ]
        return snapshot

    def context(self, history: bool = True) -> dict:
        return self.snapshot(include_history=history)
