"""Immutable records for episode-local revisable embodied memory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


def identifier(value: str, field_name: str = "identifier") -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a nonempty, trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} cannot contain control characters")
    return value


def identifiers(values, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a sequence of identifiers")
    result = tuple(identifier(value, field_name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} contains duplicate identifiers")
    return result


def step_number(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("step must be a nonnegative integer")
    return value


def freeze_json(value: Any) -> Any:
    """Copy JSON data into recursively immutable containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Evidence payload contains a nonfinite number")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Evidence payload keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item) for item in value)
    raise ValueError(f"Unsupported evidence payload type: {type(value).__name__}")


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


class Truth(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class AssertionStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class Predicate:
    name: str
    args: tuple[str, ...]

    def __post_init__(self):
        identifier(self.name, "predicate name")
        if isinstance(self.args, str):
            raise ValueError("predicate arguments must be a sequence")
        object.__setattr__(
            self,
            "args",
            tuple(identifier(arg, "predicate argument") for arg in self.args),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "args": list(self.args)}

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    step: int
    source: str
    object_ids: tuple[str, ...] = ()
    frame_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.id, "evidence id")
        step_number(self.step)
        identifier(self.source, "evidence source")
        object.__setattr__(
            self, "object_ids", identifiers(self.object_ids, "evidence object ids")
        )
        if self.frame_id is not None:
            identifier(self.frame_id, "frame id")
        if not isinstance(self.payload, Mapping):
            raise ValueError("evidence payload must be a mapping")
        object.__setattr__(self, "payload", freeze_json(self.payload))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "source": self.source,
            "object_ids": list(self.object_ids),
            "frame_id": self.frame_id,
            "payload": json_value(self.payload),
        }


@dataclass(frozen=True, slots=True)
class Assertion:
    id: str
    predicate: Predicate
    truth: Truth
    evidence_ids: tuple[str, ...]
    confidence: float
    parents: tuple[str, ...]
    status: AssertionStatus
    step: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "predicate": self.predicate.to_dict(),
            "truth": self.truth.value,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "parents": list(self.parents),
            "status": self.status.value,
            "step": self.step,
        }


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    id: str
    label: str | None
    aliases: tuple[str, ...]
    initial_position: tuple[float, float, float] | None
    last_trusted_position: tuple[float, float, float] | None
    visible: bool
    identity_confident: bool
    observed_step: int
    trusted_step: int | None
    evidence_ids: tuple[str, ...]
    trusted_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "aliases": list(self.aliases),
            "initial_position": json_value(self.initial_position),
            "last_trusted_position": json_value(self.last_trusted_position),
            "visible": self.visible,
            "identity_confident": self.identity_confident,
            "observed_step": self.observed_step,
            "trusted_step": self.trusted_step,
            "evidence_ids": list(self.evidence_ids),
            "trusted_evidence_ids": list(self.trusted_evidence_ids),
        }
