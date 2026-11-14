"""Ordered task graphs and validation of local repair fragments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from vlm_orchestrator.remm_memory.types import Predicate, identifier, identifiers

SKILLS = {
    "pick": ("holding", "Pick up {0}."),
    "place_on": ("on", "Place {0} on {1}."),
    "place_in": ("in", "Place {0} in {1}."),
}
ARITY = {"holding": 1, "on": 2, "in": 2}
START = "__start__"
GOAL = "__goal__"


def _conditions(items, object_ids, *, required=False):
    if not isinstance(items, list) or (required and not items):
        raise ValueError("conditions must be a list; effects and final conditions cannot be empty")
    predicates = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("condition must contain name and args")
        name = identifier(item.get("name"), "predicate name")
        args = item.get("args")
        if not isinstance(args, list) or len(args) != ARITY.get(name):
            raise ValueError("only holding, on and in are supported")
        args = identifiers(args, "predicate argument")
        if not set(args) <= object_ids:
            raise ValueError("condition references an undeclared object")
        predicates.append(Predicate(name, args))
    if len(set(predicates)) != len(predicates):
        raise ValueError("duplicate condition")
    return tuple(predicates)


def conflicts(left: Predicate, right: Predicate) -> bool:
    """Conservative single-gripper, single-placement symbolic model."""
    return left != right and (
        left.args[0] == right.args[0] or left.name == right.name == "holding"
    )


@dataclass(frozen=True, slots=True)
class TaskNode:
    id: str
    skill: str
    instruction: str
    object_ids: tuple[str, ...]
    preconditions: tuple[Predicate, ...]
    expected_effects: tuple[Predicate, ...]

    @classmethod
    def parse(cls, item, objects):
        if not isinstance(item, dict):
            raise ValueError("task node must be an object")
        node_id = identifier(item.get("id"), "node ID")
        if node_id in {START, GOAL}:
            raise ValueError("reserved node ID")
        skill = identifier(item.get("skill"), "skill")
        if skill not in SKILLS:
            raise ValueError(f"unsupported skill: {skill}")
        effect_name, template = SKILLS[skill]
        object_ids = item.get("object_ids")
        if not isinstance(object_ids, list) or not object_ids:
            raise ValueError("node must declare its object IDs")
        refs = identifiers(object_ids, "node object ID")
        if not set(refs) <= objects.keys():
            raise ValueError("node references an undeclared object")
        pre = _conditions(item.get("preconditions"), set(refs))
        effects = _conditions(item.get("expected_effects"), set(refs), required=True)
        if len(effects) != 1 or effects[0].name != effect_name:
            raise ValueError("each VLA skill must declare one matching expected effect")
        if any(conflicts(a, b) for a in pre for b in pre):
            raise ValueError("node has incompatible preconditions")
        instruction = template.format(*(objects[arg] for arg in effects[0].args))
        return cls(node_id, skill, instruction, refs, pre, effects)

    def to_dict(self):
        return {
            "id": self.id, "skill": self.skill, "instruction": self.instruction,
            "object_ids": list(self.object_ids),
            "preconditions": [p.to_dict() for p in self.preconditions],
            "expected_effects": [p.to_dict() for p in self.expected_effects],
        }


@dataclass(frozen=True, slots=True)
class CausalLink:
    producer: str
    consumer: str
    condition: Predicate

    def to_dict(self):
        return {
            "producer": self.producer, "consumer": self.consumer,
            "condition": self.condition.to_dict(),
        }


def _nodes(payload, objects, limit):
    if not isinstance(payload, list) or not 1 <= len(payload) <= limit:
        raise ValueError(f"expected 1 to {limit} task nodes")
    nodes = tuple(TaskNode.parse(item, objects) for item in payload)
    identifiers((node.id for node in nodes), "node ID")
    return nodes


def _apply_effects(facts, effects):
    return {fact for fact in facts if not any(conflicts(fact, p) for p in effects)} | set(effects)


@dataclass(frozen=True, slots=True)
class TaskGraph:
    objects: tuple[tuple[str, str], ...]
    nodes: tuple[TaskNode, ...]
    final_conditions: tuple[Predicate, ...]
    version: int = 1

    @classmethod
    def parse(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("plan must be a JSON object")
        items = payload.get("objects")
        if not isinstance(items, list) or not 1 <= len(items) <= 32:
            raise ValueError("expected 1 to 32 grounded objects")
        objects = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("object must contain id and description")
            object_id = identifier(item.get("id"), "object ID")
            if object_id in objects:
                raise ValueError("duplicate object ID")
            description = identifier(item.get("description"), "object description")
            if len(description.split()) > 9 or any(mark in description for mark in ",;\n"):
                raise ValueError("object description must be a short appearance-only noun phrase")
            objects[object_id] = description
        refs = set(objects)
        nodes = _nodes(payload.get("nodes"), objects, limit=32)
        final = _conditions(payload.get("final_conditions"), refs, required=True)
        if any(conflicts(a, b) for a in final for b in final):
            raise ValueError("final conditions are incompatible")
        facts = set()
        for node in nodes:
            if any(conflicts(p, known) for p in node.preconditions for known in facts):
                raise ValueError(f"earlier effects contradict preconditions of {node.id}")
            facts = _apply_effects(facts, node.expected_effects)
        if any(conflicts(p, known) for p in final for known in facts):
            raise ValueError("modeled final effects contradict the task goal")
        return cls(tuple(objects.items()), nodes, final)

    @property
    def predicates(self):
        return frozenset(self.final_conditions).union(
            p for node in self.nodes for p in (*node.preconditions, *node.expected_effects)
        )

    @property
    def links(self):
        producers = {}
        links = []
        for node in self.nodes:
            links.extend(CausalLink(producers.get(p, START), node.id, p) for p in node.preconditions)
            effects = node.expected_effects
            producers = {p: source for p, source in producers.items() if not any(conflicts(p, e) for e in effects)}
            producers.update((p, node.id) for p in effects)
        links.extend(CausalLink(producers.get(p, START), GOAL, p) for p in self.final_conditions)
        return tuple(links)

    def affected(self, required, pending):
        """Trace affected pending consumers, including their downstream nodes."""
        affected = set()
        for link in self.links:
            if link.consumer in pending and (link.condition in required or link.producer in affected):
                affected.add(link.consumer)
        return affected

    def insert_repair(self, payload, cursor, facts, required, protected):
        """Check a repair and the retained continuation from verified facts."""
        repair = _nodes(payload, dict(self.objects), limit=8)
        if {node.id for node in repair} & {node.id for node in self.nodes}:
            raise ValueError("repair must use new node IDs")
        patched = replace(self, nodes=(*self.nodes[:cursor], *repair, *self.nodes[cursor:]),
                          version=self.version + 1)
        patched.validate_repair(cursor, len(repair), facts, required, protected)
        return patched, len(repair)

    def validate_repair(self, cursor, repair_count, facts, required, protected):
        if not protected <= facts:
            raise ValueError("protected effects need fresh verification")
        for index, node in enumerate(self.nodes[cursor:], start=1):
            if not set(node.expected_effects) <= facts:
                if not set(node.preconditions) <= facts:
                    raise ValueError(f"unsatisfied symbolic preconditions for {node.id}")
                facts = _apply_effects(facts, node.expected_effects)
            if index == repair_count and not (set(required) | protected) <= facts:
                raise ValueError("repair must restore required conditions and preserve verified effects")
        if not set(self.final_conditions) <= facts:
            raise ValueError("repaired continuation cannot establish final conditions")

    def to_dict(self):
        return {
            "version": self.version,
            "objects": [{"id": key, "description": value} for key, value in self.objects],
            "nodes": [node.to_dict() for node in self.nodes],
            "final_conditions": [p.to_dict() for p in self.final_conditions],
            "links": [link.to_dict() for link in self.links],
        }
