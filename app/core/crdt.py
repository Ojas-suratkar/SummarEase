"""
crdt.py -- the pure, offline-first merge core for SummarEase.

WHY THIS MODULE EXISTS
======================
SummarEase must be usable with no network at all: on a plane, on a train, in a
lift. The user reads, highlights, annotates, re-tags and deletes entries while
disconnected -- potentially on several devices at once -- and when connectivity
comes back every device must end up showing *the same thing*, without anybody's
work being silently thrown away.

The tempting shortcut is "last write wins on the whole record": each device
stamps the row with a timestamp, the newest row overwrites the older one. That
approach is data loss dressed up as a merge strategy:

  * It is whole-record. If the phone changed only `title` and the laptop changed
    only `notes`, one of those two edits vanishes, even though the two edits do
    not conflict at all.
  * It trusts wall clocks. Device clocks drift, users change timezones, a device
    that has been off for a week comes back with a stale clock, and a malicious
    or buggy client can simply claim the year 2099 and win every future merge
    forever.
  * It is not idempotent or commutative. Replay a message, or receive two
    messages out of order, and you get a different answer. Sync pipelines
    (service workers, retry queues, at-least-once delivery) replay and reorder
    constantly.

A CRDT (Conflict-free Replicated Data Type) fixes this by construction. Every
piece of state is a *join semilattice*: merging is

  * commutative   -- merge(a, b) == merge(b, a)          (arrival order is irrelevant)
  * associative   -- merge(merge(a, b), c) == merge(a, merge(b, c))
  * idempotent    -- merge(a, a) == a                     (duplicates are harmless)

The consequence -- the whole point -- is **strong eventual consistency**: any
two replicas that have observed the same *set* of operations are in the same
state, regardless of the order in which they observed them, regardless of
duplicates, with no coordination, no locking, and no server arbitration.

DESIGN OF THE SUMMAREASE DOCUMENT CRDT
======================================
A SummarEase entity (an entry, a note, a tag, an annotation) is modelled as a
*map of independently mergeable fields*, not as one opaque blob:

  * scalar fields (`title`, `summary`, `body`, `rating`, ...)  -> LWWRegister
  * set-valued fields (`tags`, `collaborators`, ...)           -> ORSet
  * append-only fields (audit trails, read history)            -> GSet
  * the entity itself                                          -> monotone delete tombstone

Because fields merge independently, the phone's `title` edit and the laptop's
`notes` edit both survive. Only two *genuinely concurrent writes to the same
field* need a winner, and that is what the deterministic tie-break below is for.

TIE-BREAKING: (lamport, replica_id, op_id) -- NEVER wall clock
--------------------------------------------------------------
Every operation carries a Lamport logical clock value. Lamport clocks respect
causality: if op A could have influenced op B, then A.lamport < B.lamport. So
when one edit genuinely happened after (and with knowledge of) another, the
later one wins -- which is what a human expects.

When two ops are *concurrent* (neither saw the other) their Lamport values may
tie. We then break the tie on `replica_id`, and finally on `op_id`. Both are
opaque, stable identifiers, so the comparison is a total order that every
replica computes identically from the operation payload alone. That is what
makes the merge deterministic rather than merely "eventually something".

`wall_clock` is carried along **only** as a human-facing hint ("edited about 5
minutes ago", ordering an activity feed). It is never consulted for merge
correctness, because a skewed, wrong or dishonest clock would then be able to
corrupt or capture state permanently. Logical time cannot be gamed into the
future in a way that survives: a replica can inflate its own Lamport counter,
but it cannot retroactively lose a causal edit, and other replicas' later
(causally aware) edits will still dominate.

Stdlib only. No Flask, no SQLAlchemy, no third-party packages: this file is
pure algorithm so it can be unit-tested in isolation and, if we ever want to,
transliterated to JavaScript for the service worker.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Operation",
    "VectorClock",
    "LWWRegister",
    "GSet",
    "ORSet",
    "ReplicaState",
    "merge_operations",
    "resolve",
    "diff_for_sync",
    "new_op_id",
    "canonical_key",
]

# A sort key is (lamport, replica_id, op_id): a total order over operations that
# every replica derives from the operation itself, with no shared state.
SortKey = Tuple[int, str, str]

# Field kinds understood by ReplicaState's optional schema.
KIND_LWW = "lww"
KIND_ORSET = "orset"
KIND_GSET = "gset"


def new_op_id() -> str:
    """A globally unique, client-generated operation id (uuid4 hex).

    Client-generated on purpose: an offline device must be able to mint ids with
    no server round-trip, and the id doubles as the ORSet's unique add-tag and
    as the dedupe key that makes replay idempotent.
    """
    return uuid.uuid4().hex


def canonical_key(value: Any) -> str:
    """Stable, hashable identity for an arbitrary JSON-able set element.

    Set CRDTs need to compare elements for equality across replicas and across
    the JSON wire. `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are the same
    element, so we key on canonical JSON (sorted keys, tight separators). Long
    elements are hashed to keep in-memory keys small; the original value is kept
    alongside so `value()` can return it untouched.
    """
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if len(raw) <= 128:
        return raw
    return "#" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _key_to_list(key: Optional[SortKey]) -> Optional[List[Any]]:
    return None if key is None else [key[0], key[1], key[2]]


def _key_from_list(raw: Optional[Sequence[Any]]) -> Optional[SortKey]:
    if raw is None:
        return None
    return (int(raw[0]), str(raw[1]), str(raw[2]))


def _max_key(a: Optional[SortKey], b: Optional[SortKey]) -> Optional[SortKey]:
    """Join of two sort keys, treating `None` as bottom (-infinity)."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------


@dataclass
class Operation:
    """One immutable, replicated intent.

    Operations are the only thing that travels between devices. They are
    immutable facts ("replica R, at logical time L, set entry E's title to X"),
    never mutable rows, which is what allows at-least-once delivery, arbitrary
    reordering, and replay to be safe.

    Fields
    ------
    op_id      : globally unique (uuid4 hex). Dedupe key *and* ORSet add-tag.
    replica_id : which device produced it. Part of the tie-break.
    entity     : entity kind -- "entry" | "note" | "tag" | "annotation" | ...
    entity_id  : which entity this op touches.
    field      : which field ("" for whole-entity ops such as `delete`).
    action     : "set" | "add" | "remove" | "delete".
    value      : payload. For `remove` this may be the bare element, or
                 `{"element": <elem>, "tags": [<add op_ids>]}` for a precise
                 observed-remove (see ORSet).
    lamport    : Lamport logical clock at the producing replica. Ordering.
    wall_clock : unix seconds. HUMAN HINT ONLY -- never used for correctness,
                 because device clocks are skewed, resettable and forgeable.
    """

    op_id: str
    replica_id: str
    entity: str
    entity_id: str
    field: str
    action: str
    value: Any
    lamport: int
    wall_clock: float = dc_field(default_factory=time.time)

    # -- ordering ----------------------------------------------------------
    def sort_key(self) -> SortKey:
        """The deterministic total order: (lamport, replica_id, op_id).

        Lamport first, so a causally later edit always beats the edit it saw.
        Then replica_id, then op_id, to settle genuine concurrency the same way
        on every device in the fleet. Deliberately NOT wall_clock.
        """
        return (self.lamport, self.replica_id, self.op_id)

    def __lt__(self, other: "Operation") -> bool:
        return self.sort_key() < other.sort_key()

    # -- transport ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "replica_id": self.replica_id,
            "entity": self.entity,
            "entity_id": self.entity_id,
            "field": self.field,
            "action": self.action,
            "value": self.value,
            "lamport": self.lamport,
            "wall_clock": self.wall_clock,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Operation":
        return cls(
            op_id=str(raw["op_id"]),
            replica_id=str(raw["replica_id"]),
            entity=str(raw.get("entity", "")),
            entity_id=str(raw["entity_id"]),
            field=str(raw.get("field", "")),
            action=str(raw["action"]),
            value=raw.get("value"),
            lamport=int(raw["lamport"]),
            wall_clock=float(raw.get("wall_clock", 0.0)),
        )


# ---------------------------------------------------------------------------
# VectorClock
# ---------------------------------------------------------------------------


class VectorClock:
    """Per-replica counters used to answer "what have you already seen?".

    We store, per replica, the highest Lamport value we have observed *from
    that replica*. Because a replica's own ops carry strictly increasing Lamport
    values, "R -> 41" is exactly equivalent to "I have every op R produced up to
    logical time 41", which is all the sync protocol needs (see
    `ReplicaState.ops_since` / `diff_for_sync`).

    The componentwise partial order gives us causality comparison:
      before     -- self <= other everywhere and differs somewhere
      after      -- the mirror image
      equal      -- identical
      concurrent -- neither dominates: the two replicas diverged
    "concurrent" is the interesting case; it is where tie-breaking applies.
    """

    __slots__ = ("_counters",)

    def __init__(self, counters: Optional[Dict[str, int]] = None) -> None:
        self._counters: Dict[str, int] = dict(counters or {})

    # -- basics ------------------------------------------------------------
    def get(self, replica_id: str) -> int:
        return self._counters.get(replica_id, 0)

    def max_value(self) -> int:
        return max(self._counters.values()) if self._counters else 0

    def observe(self, replica_id: str, lamport: int) -> None:
        """Record that we have seen `replica_id`'s op at logical time `lamport`.

        Monotone (max), so observing the same op twice is a no-op: idempotency
        at the clock level, not just at the state level.
        """
        if lamport > self._counters.get(replica_id, 0):
            self._counters[replica_id] = lamport

    def increment(self, replica_id: str) -> int:
        """Advance `replica_id`'s entry past everything we know, return the value.

        We use `max(all entries) + 1` rather than `own + 1` so that the entry is
        a genuine Lamport timestamp: an op minted here is ordered strictly after
        every op this replica has already observed, which is precisely the
        causality guarantee the tie-break relies on.
        """
        nxt = max(self.max_value(), self._counters.get(replica_id, 0)) + 1
        self._counters[replica_id] = nxt
        return nxt

    # -- lattice -----------------------------------------------------------
    def merge(self, other: "VectorClock") -> "VectorClock":
        """Componentwise max -- the join. Returns a NEW clock (no mutation)."""
        merged = dict(self._counters)
        for replica_id, counter in other._counters.items():
            if counter > merged.get(replica_id, 0):
                merged[replica_id] = counter
        return VectorClock(merged)

    def compare(self, other: "VectorClock") -> str:
        """Return "before" | "after" | "equal" | "concurrent"."""
        self_dominates = False
        other_dominates = False
        for replica_id in set(self._counters) | set(other._counters):
            mine = self.get(replica_id)
            theirs = other.get(replica_id)
            if mine > theirs:
                self_dominates = True
            elif mine < theirs:
                other_dominates = True
        if self_dominates and other_dominates:
            return "concurrent"
        if self_dominates:
            return "after"
        if other_dominates:
            return "before"
        return "equal"

    def covers(self, op: Operation) -> bool:
        """True if this clock already accounts for `op`."""
        return self.get(op.replica_id) >= op.lamport

    # -- transport ---------------------------------------------------------
    def to_dict(self) -> Dict[str, int]:
        return dict(self._counters)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "VectorClock":
        return cls({str(k): int(v) for k, v in (raw or {}).items()})

    def copy(self) -> "VectorClock":
        return VectorClock(self._counters)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VectorClock) and self._counters == other._counters

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"VectorClock({self._counters!r})"


# ---------------------------------------------------------------------------
# LWWRegister
# ---------------------------------------------------------------------------


class LWWRegister:
    """Last-writer-wins register for a single scalar field.

    "Last" means *last in logical time*, not last by wall clock. The register
    keeps the value whose `(lamport, replica_id, op_id)` key is greatest. Since
    `max` over a total order is commutative, associative and idempotent, this is
    a join semilattice and therefore convergent.

    LWW *is* lossy for a single field -- one of two concurrent edits to the same
    field must lose. That is unavoidable without user-facing conflict UI. The
    crucial difference from record-level LWW is granularity: only the one
    contended field picks a winner, while every other field on the entity keeps
    both users' work.
    """

    __slots__ = ("_value", "_key")

    def __init__(self, value: Any = None, key: Optional[SortKey] = None) -> None:
        self._value = value
        self._key = key

    def apply(self, op: Operation) -> None:
        key = op.sort_key()
        if self._key is None or key > self._key:
            self._value = op.value
            self._key = key

    def value(self) -> Any:
        return self._value

    def key(self) -> Optional[SortKey]:
        return self._key

    def is_empty(self) -> bool:
        return self._key is None

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        if other._key is None:
            return LWWRegister(self._value, self._key)
        if self._key is None or other._key > self._key:
            return LWWRegister(other._value, other._key)
        return LWWRegister(self._value, self._key)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": KIND_LWW, "value": self._value, "key": _key_to_list(self._key)}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LWWRegister":
        return cls(raw.get("value"), _key_from_list(raw.get("key")))


# ---------------------------------------------------------------------------
# GSet
# ---------------------------------------------------------------------------


class GSet:
    """Grow-only set: elements can be added, never removed.

    The simplest CRDT there is -- merge is set union, which is trivially
    commutative/associative/idempotent. We use it for genuinely append-only
    SummarEase data such as read-history and audit breadcrumbs, where "remove"
    is not a meaningful operation and a tombstone would be pure overhead.
    """

    __slots__ = ("_elements",)

    def __init__(self, elements: Optional[Dict[str, Any]] = None) -> None:
        # canonical_key -> original value
        self._elements: Dict[str, Any] = dict(elements or {})

    def apply(self, op: Operation) -> None:
        if op.action != "add":
            # A grow-only set cannot honour a remove. Ignoring it (rather than
            # raising) keeps a misrouted op from wedging an offline client's
            # whole replay queue; the schema is what decides the field kind.
            return
        self._elements[canonical_key(op.value)] = op.value

    def add(self, element: Any) -> None:
        self._elements[canonical_key(element)] = element

    def value(self) -> List[Any]:
        return [self._elements[k] for k in sorted(self._elements)]

    def merge(self, other: "GSet") -> "GSet":
        merged = dict(self._elements)
        merged.update(other._elements)
        return GSet(merged)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": KIND_GSET, "elements": self._elements}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GSet":
        return cls(dict(raw.get("elements") or {}))


# ---------------------------------------------------------------------------
# ORSet
# ---------------------------------------------------------------------------


class ORSet:
    """Observed-Remove set -- the right CRDT for `tags`, `collaborators`, ...

    THE CLASSIC BUG, AND HOW WE AVOID IT
    ------------------------------------
    A naive set CRDT stores bare elements plus a "removed" flag, and then:

      * a duplicated or late-arriving `add "urgent"` **resurrects** a tag the
        user deleted ten minutes ago (the sync queue retried; the tag is back),
        or
      * a "remove wins forever" rule makes the tag **undeletable**: the user can
        never legitimately re-add "urgent" again.

    OR-Set separates the two cases by giving every *add* a unique tag -- here,
    the add op's `op_id`, which is uuid4 and therefore never reused. The set
    stores (element, add-tag) pairs; a remove stores tombstones for the add-tags
    it *observed*. An element is present iff it has at least one add-tag that no
    remove has tombstoned.

      * Re-delivering the very same add is a no-op: its tag is already
        tombstoned, so the element stays removed. No resurrection.
      * A genuinely NEW add after the remove mints a FRESH tag which no remove
        has ever seen, so the element comes back. Re-adds work.

    That distinction -- "is this the same add I already deleted, or a new one?"
    -- is exactly what identity-per-add buys us, and it is invisible to
    timestamp-based schemes.

    REMOVES WITHOUT EXPLICIT TAGS
    -----------------------------
    A well-behaved client sends `value = {"element": e, "tags": [add op_ids]}`,
    listing the adds it observed (use `ReplicaState.observed_tags`). For
    lenient/legacy clients that send just the bare element we fall back to a
    *causal threshold*: the remove also tombstones every add whose sort key is
    below the remove's own sort key. That is still a pure function of the op set
    (it compares keys, never arrival order), so it stays commutative,
    associative and idempotent -- and a later re-add, having a strictly greater
    Lamport value, still survives.
    """

    __slots__ = ("_elements",)

    def __init__(self, elements: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        # canonical_key -> {"value": Any,
        #                   "adds": {tag: SortKey},
        #                   "tombs": set[tag],
        #                   "bound": Optional[SortKey]}
        self._elements: Dict[str, Dict[str, Any]] = elements or {}

    # -- internals ---------------------------------------------------------
    def _entry(self, ekey: str, value: Any) -> Dict[str, Any]:
        entry = self._elements.get(ekey)
        if entry is None:
            entry = {"value": value, "adds": {}, "tombs": set(), "bound": None}
            self._elements[ekey] = entry
        return entry

    @staticmethod
    def _split_remove(value: Any) -> Tuple[Any, Optional[List[str]]]:
        """Decode a remove payload into (element, explicit tags or None)."""
        if isinstance(value, dict) and "element" in value and "tags" in value:
            tags = value.get("tags") or []
            if isinstance(tags, (list, tuple, set)):
                return value["element"], [str(t) for t in tags]
        return value, None

    # -- CRDT ops ----------------------------------------------------------
    def apply(self, op: Operation) -> None:
        key = op.sort_key()
        if op.action == "add":
            entry = self._entry(canonical_key(op.value), op.value)
            # The add-tag is the op_id: unique per add, never recycled, so a
            # replayed add carries the tag that was already tombstoned.
            entry["adds"][op.op_id] = key
        elif op.action in ("remove", "delete"):
            element, tags = self._split_remove(op.value)
            entry = self._entry(canonical_key(element), element)
            if tags is not None:
                entry["tombs"].update(tags)
            else:
                entry["bound"] = _max_key(entry["bound"], key)

    def tags_for(self, element: Any) -> List[str]:
        """The add-tags currently making `element` visible.

        A client calls this when it wants to emit a precise observed-remove:
        it removes exactly what it can see, and nothing it has not seen.
        """
        entry = self._elements.get(canonical_key(element))
        if entry is None:
            return []
        return sorted(t for t, k in entry["adds"].items() if self._tag_live(entry, t, k))

    @staticmethod
    def _tag_live(entry: Dict[str, Any], tag: str, key: SortKey) -> bool:
        if tag in entry["tombs"]:
            return False
        bound = entry["bound"]
        return bound is None or key > bound

    def contains(self, element: Any) -> bool:
        entry = self._elements.get(canonical_key(element))
        if entry is None:
            return False
        return any(self._tag_live(entry, t, k) for t, k in entry["adds"].items())

    def value(self) -> List[Any]:
        """Live elements, in canonical-key order so snapshots compare equal."""
        out: List[Any] = []
        for ekey in sorted(self._elements):
            entry = self._elements[ekey]
            if any(self._tag_live(entry, t, k) for t, k in entry["adds"].items()):
                out.append(entry["value"])
        return out

    def merge(self, other: "ORSet") -> "ORSet":
        """Join: union of adds, union of tombstones, max of causal bounds."""
        merged: Dict[str, Dict[str, Any]] = {}
        for ekey in set(self._elements) | set(other._elements):
            mine = self._elements.get(ekey)
            theirs = other._elements.get(ekey)
            if mine is None:
                base = theirs
                assert base is not None
                merged[ekey] = {
                    "value": base["value"],
                    "adds": dict(base["adds"]),
                    "tombs": set(base["tombs"]),
                    "bound": base["bound"],
                }
                continue
            if theirs is None:
                merged[ekey] = {
                    "value": mine["value"],
                    "adds": dict(mine["adds"]),
                    "tombs": set(mine["tombs"]),
                    "bound": mine["bound"],
                }
                continue
            adds = dict(mine["adds"])
            adds.update(theirs["adds"])
            merged[ekey] = {
                "value": mine["value"],
                "adds": adds,
                "tombs": set(mine["tombs"]) | set(theirs["tombs"]),
                "bound": _max_key(mine["bound"], theirs["bound"]),
            }
        return ORSet(merged)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": KIND_ORSET,
            "elements": {
                ekey: {
                    "value": entry["value"],
                    "adds": {tag: _key_to_list(k) for tag, k in entry["adds"].items()},
                    "tombs": sorted(entry["tombs"]),
                    "bound": _key_to_list(entry["bound"]),
                }
                for ekey, entry in self._elements.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ORSet":
        elements: Dict[str, Dict[str, Any]] = {}
        for ekey, entry in (raw.get("elements") or {}).items():
            elements[ekey] = {
                "value": entry.get("value"),
                "adds": {
                    str(tag): _key_from_list(k) for tag, k in (entry.get("adds") or {}).items()
                },
                "tombs": set(entry.get("tombs") or []),
                "bound": _key_from_list(entry.get("bound")),
            }
        return cls(elements)


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class _Entity:
    """One SummarEase entity: a map of independently mergeable fields.

    Also carries the **delete tombstone**. Deletion is modelled as a monotone
    flag (once set by any replica, it stays set) rather than as row removal.
    If we actually dropped the row, the next sync with a replica that still has
    the entity would happily "re-create" it and the user's delete would be
    undone -- the zombie-record bug every naive sync implementation ships with.
    A tombstone instead *propagates*: it is itself a fact that spreads.

    Field ops that arrive after (or concurrently with) the delete are still
    recorded, because the op log keeps them; they are simply not surfaced while
    the entity is deleted. That keeps merge associative and leaves the door open
    for an undelete feature later without having lost the data.
    """

    __slots__ = (
        "entity",
        "entity_key",
        "registers",
        "orsets",
        "gsets",
        "deleted_key",
        "last_wall",
    )

    def __init__(self, entity: str = "", entity_key: Optional[SortKey] = None) -> None:
        # The entity *kind* is pinned by the causally FIRST op (smallest sort
        # key) that mentions it, not by whichever op happened to arrive first --
        # otherwise two replicas that saw the same ops in different orders could
        # disagree about the kind, which would break convergence.
        self.entity = entity
        self.entity_key = entity_key
        self.registers: Dict[str, LWWRegister] = {}
        self.orsets: Dict[str, ORSet] = {}
        self.gsets: Dict[str, GSet] = {}
        self.deleted_key: Optional[SortKey] = None
        self.last_wall: float = 0.0

    # -- apply -------------------------------------------------------------
    def apply(self, op: Operation, kind: str) -> None:
        if op.entity and (self.entity_key is None or op.sort_key() < self.entity_key):
            self.entity = op.entity
            self.entity_key = op.sort_key()
        if op.wall_clock > self.last_wall:
            self.last_wall = op.wall_clock

        if op.action == "delete" and not op.field:
            # Monotone: the first delete we ever see wins forever. We keep the
            # key so tooling can say who deleted it and when (logically).
            self.deleted_key = _max_key(self.deleted_key, op.sort_key())
            return

        if kind == KIND_ORSET:
            self.orsets.setdefault(op.field, ORSet()).apply(op)
        elif kind == KIND_GSET:
            self.gsets.setdefault(op.field, GSet()).apply(op)
        else:
            self.registers.setdefault(op.field, LWWRegister()).apply(op)

    # -- lattice -----------------------------------------------------------
    def merge(self, other: "_Entity") -> "_Entity":
        if self.entity_key is None:
            out = _Entity(other.entity, other.entity_key)
        elif other.entity_key is None or self.entity_key < other.entity_key:
            out = _Entity(self.entity, self.entity_key)
        else:
            out = _Entity(other.entity, other.entity_key)
        for name in set(self.registers) | set(other.registers):
            mine = self.registers.get(name, LWWRegister())
            out.registers[name] = mine.merge(other.registers.get(name, LWWRegister()))
        for name in set(self.orsets) | set(other.orsets):
            mine_s = self.orsets.get(name, ORSet())
            out.orsets[name] = mine_s.merge(other.orsets.get(name, ORSet()))
        for name in set(self.gsets) | set(other.gsets):
            mine_g = self.gsets.get(name, GSet())
            out.gsets[name] = mine_g.merge(other.gsets.get(name, GSet()))
        out.deleted_key = _max_key(self.deleted_key, other.deleted_key)
        out.last_wall = max(self.last_wall, other.last_wall)
        return out

    # -- materialise -------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, reg in self.registers.items():
            if not reg.is_empty():
                out[name] = reg.value()
        for name, oset in self.orsets.items():
            out[name] = oset.value()
        for name, gset in self.gsets.items():
            out[name] = gset.value()
        out["_entity"] = self.entity
        out["_deleted"] = self.deleted_key is not None
        # `_updated_at` is derived by max() over the op set, so it is still
        # deterministic -- but it is a display hint, never a merge input.
        out["_updated_at"] = self.last_wall
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_key": _key_to_list(self.entity_key),
            "registers": {k: v.to_dict() for k, v in self.registers.items()},
            "orsets": {k: v.to_dict() for k, v in self.orsets.items()},
            "gsets": {k: v.to_dict() for k, v in self.gsets.items()},
            "deleted_key": _key_to_list(self.deleted_key),
            "last_wall": self.last_wall,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "_Entity":
        ent = cls(str(raw.get("entity", "")), _key_from_list(raw.get("entity_key")))
        ent.registers = {
            k: LWWRegister.from_dict(v) for k, v in (raw.get("registers") or {}).items()
        }
        ent.orsets = {k: ORSet.from_dict(v) for k, v in (raw.get("orsets") or {}).items()}
        ent.gsets = {k: GSet.from_dict(v) for k, v in (raw.get("gsets") or {}).items()}
        ent.deleted_key = _key_from_list(raw.get("deleted_key"))
        ent.last_wall = float(raw.get("last_wall", 0.0))
        return ent


# ---------------------------------------------------------------------------
# ReplicaState
# ---------------------------------------------------------------------------


class ReplicaState:
    """The full merged state of one replica: a map of entity_id -> fields.

    Holds three things:
      * the **op log** (`op_id -> Operation`), which is the source of truth and
        what makes dedupe, `ops_since` and sync possible;
      * the **materialised CRDT structures**, so reads are cheap;
      * a **vector clock** summarising everything observed, so a peer can ask
        "what am I missing?" in one small JSON object instead of shipping ids.

    Field kind is chosen by the optional `field_types` schema, keyed either as
    "<entity>.<field>" or bare "<field>"; otherwise it is inferred from the
    action ("set" -> LWW, "add"/"remove" -> ORSet). The schema exists so that
    genuinely append-only fields can opt into the cheaper GSet.
    """

    def __init__(
        self,
        replica_id: str,
        field_types: Optional[Dict[str, str]] = None,
    ) -> None:
        self.replica_id = replica_id
        self.field_types: Dict[str, str] = dict(field_types or {})
        self.clock = VectorClock()
        self.lamport: int = 0
        self._ops: Dict[str, Operation] = {}
        self._entities: Dict[str, _Entity] = {}

    # -- schema ------------------------------------------------------------
    def _kind(self, op: Operation) -> str:
        explicit = self.field_types.get(f"{op.entity}.{op.field}") or self.field_types.get(
            op.field
        )
        if explicit in (KIND_LWW, KIND_ORSET, KIND_GSET):
            return explicit
        if op.action in ("add", "remove"):
            return KIND_ORSET
        return KIND_LWW

    # -- authoring ---------------------------------------------------------
    def next_lamport(self) -> int:
        """Lamport tick for a locally authored op: strictly after all we know."""
        self.lamport = max(self.lamport, self.clock.max_value()) + 1
        return self.lamport

    def make_op(
        self,
        entity: str,
        entity_id: str,
        field: str,
        action: str,
        value: Any = None,
        wall_clock: Optional[float] = None,
    ) -> Operation:
        """Mint a locally authored operation and apply it here immediately."""
        op = Operation(
            op_id=new_op_id(),
            replica_id=self.replica_id,
            entity=entity,
            entity_id=entity_id,
            field=field,
            action=action,
            value=value,
            lamport=self.next_lamport(),
            wall_clock=time.time() if wall_clock is None else wall_clock,
        )
        self.apply(op)
        return op

    def observed_tags(self, entity_id: str, field: str, element: Any) -> List[str]:
        """Add-tags this replica can currently see for an ORSet element.

        Used to build a precise observed-remove: `{"element": e, "tags": ...}`.
        """
        ent = self._entities.get(entity_id)
        if ent is None or field not in ent.orsets:
            return []
        return ent.orsets[field].tags_for(element)

    def make_remove_op(
        self, entity: str, entity_id: str, field: str, element: Any
    ) -> Operation:
        """Convenience: an observed-remove for `element` in `entity_id.field`."""
        tags = self.observed_tags(entity_id, field, element)
        return self.make_op(
            entity, entity_id, field, "remove", {"element": element, "tags": tags}
        )

    # -- applying ----------------------------------------------------------
    def apply(self, op: Operation) -> bool:
        """Apply one op. Returns False if it was already seen (idempotent).

        Dedupe is by `op_id`, so at-least-once delivery -- a retried service
        worker POST, a duplicated websocket frame, a replayed IndexedDB queue --
        cannot double-apply anything.
        """
        if op.op_id in self._ops:
            return False
        self._ops[op.op_id] = op
        self.clock.observe(op.replica_id, op.lamport)
        if op.lamport > self.lamport:
            self.lamport = op.lamport
        ent = self._entities.get(op.entity_id)
        if ent is None:
            ent = _Entity(op.entity)
            self._entities[op.entity_id] = ent
        ent.apply(op, self._kind(op))
        return True

    def apply_many(self, ops: List[Operation]) -> int:
        """Apply a batch, returning how many were new."""
        return sum(1 for op in ops if self.apply(op))

    def merge(self, other: "ReplicaState") -> None:
        """Merge another replica into this one, in place.

        Implemented as a union of op logs: every op the peer has and we do not
        gets applied here. Because each underlying structure's `apply` is a
        lattice join, replaying in whatever order the union happens to produce
        yields the same state -- which is what makes `merge` commutative and
        associative at the ReplicaState level too.
        """
        for op in other._ops.values():
            self.apply(op)

    # -- reading -----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Materialise current state: {entity_id: {field: value, "_deleted": ...}}."""
        return {eid: ent.snapshot() for eid, ent in sorted(self._entities.items())}

    def live_snapshot(self) -> Dict[str, Any]:
        """Snapshot with tombstoned entities filtered out (what the UI renders)."""
        return {
            eid: state
            for eid, state in self.snapshot().items()
            if not state.get("_deleted")
        }

    def ops(self) -> List[Operation]:
        """The whole op log in deterministic order."""
        return sorted(self._ops.values(), key=Operation.sort_key)

    def ops_since(self, clock: VectorClock) -> List[Operation]:
        """Ops this replica holds that `clock` has not yet observed.

        Cheap and exact: an op is missing iff its Lamport value exceeds the
        peer's entry for its originating replica, because a replica's own ops
        carry strictly increasing Lamport values.
        """
        return sorted(
            (op for op in self._ops.values() if not clock.covers(op)),
            key=Operation.sort_key,
        )

    def has_op(self, op_id: str) -> bool:
        return op_id in self._ops

    def __len__(self) -> int:
        return len(self._ops)

    # -- transport ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "replica_id": self.replica_id,
            "field_types": self.field_types,
            "clock": self.clock.to_dict(),
            "lamport": self.lamport,
            "ops": [op.to_dict() for op in self.ops()],
            "entities": {eid: ent.to_dict() for eid, ent in self._entities.items()},
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ReplicaState":
        state = cls(str(raw.get("replica_id", "")), raw.get("field_types"))
        state.clock = VectorClock.from_dict(raw.get("clock"))
        state.lamport = int(raw.get("lamport", 0))
        state._ops = {
            str(o["op_id"]): Operation.from_dict(o) for o in (raw.get("ops") or [])
        }
        state._entities = {
            eid: _Entity.from_dict(e) for eid, e in (raw.get("entities") or {}).items()
        }
        return state


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def merge_operations(local: List[Operation], remote: List[Operation]) -> List[Operation]:
    """Union two op streams, de-duplicated by `op_id`, in deterministic order.

    Set union is commutative, associative and idempotent, so this is safe to
    call with overlapping batches, in any order, as many times as the retry
    queue likes. The result is sorted by `(lamport, replica_id, op_id)`, which
    is also a causal order (Lamport first), so a consumer that replays the list
    top to bottom never sees an effect before its cause.
    """
    merged: Dict[str, Operation] = {}
    for op in list(local) + list(remote):
        merged.setdefault(op.op_id, op)
    return sorted(merged.values(), key=Operation.sort_key)


def resolve(ops: List[Operation], field_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Fold an op list into final materialised state.

    Returns `{entity_id: {field: value, "_deleted": bool, ...}}`. Pure: the
    result depends only on the *set* of ops, never on their order in the list,
    which is exactly the property the server relies on when it resolves a batch
    that arrived from several devices at once.
    """
    state = ReplicaState("resolver", field_types)
    state.apply_many(list(ops))
    return state.snapshot()


def diff_for_sync(client_clock: Dict[str, int], server_ops: List[Operation]) -> List[Operation]:
    """What the server must send a client, given the client's vector clock.

    The client posts its clock (`{replica_id: max_lamport_seen}`); we return
    exactly the ops it is missing, in causal order. Over-sending is merely
    wasteful (application is idempotent), under-sending would break
    convergence -- so the test suite asserts this is *exactly* the missing set.
    """
    clock = VectorClock.from_dict(client_clock)
    return sorted(
        (op for op in server_ops if not clock.covers(op)), key=Operation.sort_key
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    _FAILURES: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            _FAILURES.append(name)

    def canon(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, default=str)

    # -- 1. Operation / VectorClock basics --------------------------------
    op = Operation(new_op_id(), "r1", "entry", "e1", "title", "set", "Hello", 3, 1.0)
    check("Operation round-trips through JSON", Operation.from_dict(json.loads(json.dumps(op.to_dict()))) == op)
    check("Operation sort key is (lamport, replica_id, op_id)", op.sort_key() == (3, "r1", op.op_id))

    a, b = VectorClock({"r1": 2}), VectorClock({"r1": 2, "r2": 1})
    check("VectorClock compare: before", a.compare(b) == "before")
    check("VectorClock compare: after", b.compare(a) == "after")
    check("VectorClock compare: equal", a.compare(VectorClock({"r1": 2})) == "equal")
    check(
        "VectorClock compare: concurrent",
        VectorClock({"r1": 3}).compare(VectorClock({"r2": 3})) == "concurrent",
    )
    check("VectorClock merge is componentwise max", a.merge(b).to_dict() == {"r1": 2, "r2": 1})
    check("VectorClock round-trips", VectorClock.from_dict(b.to_dict()) == b)
    vc = VectorClock({"r1": 5, "r2": 9})
    check("VectorClock.increment is Lamport-style (max+1)", vc.increment("r1") == 10)

    # -- 2. Idempotency ----------------------------------------------------
    s = ReplicaState("r1")
    o1 = Operation(new_op_id(), "r1", "entry", "e1", "title", "set", "A", 1, 1.0)
    o2 = Operation(new_op_id(), "r1", "entry", "e1", "tags", "add", "urgent", 2, 2.0)
    first = s.apply_many([o1, o2])
    before = canon(s.snapshot())
    again = s.apply_many([o1, o2, o1, o2])
    check("Idempotency: duplicates are rejected", first == 2 and again == 0)
    check("Idempotency: state unchanged by replay", canon(s.snapshot()) == before)
    s2 = ReplicaState("r2")
    s2.apply_many([o2, o1, o1, o2, o2])
    check("Idempotency + commutativity on 2 ops", canon(s2.snapshot()) == before)

    # -- 3. Concurrent tie-breaking determinism ---------------------------
    ca = Operation("aaa" * 10, "device-A", "entry", "e9", "title", "set", "from-A", 7, 9_999_999.0)
    cb = Operation("bbb" * 10, "device-B", "entry", "e9", "title", "set", "from-B", 7, 1.0)
    sa, sb = ReplicaState("x"), ReplicaState("y")
    sa.apply_many([ca, cb])
    sb.apply_many([cb, ca])
    check(
        "Tie-break: same result in both arrival orders",
        sa.snapshot()["e9"]["title"] == sb.snapshot()["e9"]["title"],
    )
    check(
        "Tie-break: higher replica_id wins, NOT the newer wall clock",
        sa.snapshot()["e9"]["title"] == "from-B",
    )
    cc = Operation(new_op_id(), "device-A", "entry", "e9", "title", "set", "causal", 8, 0.0)
    sa.apply(cc)
    check("Tie-break: higher lamport beats replica_id", sa.snapshot()["e9"]["title"] == "causal")
    lo = Operation("z" * 32, "device-Z", "entry", "e9", "title", "set", "stale", 2, 9e9)
    sa.apply(lo)
    check("Tie-break: a late op with an inflated wall clock cannot win", sa.snapshot()["e9"]["title"] == "causal")

    # -- 4. Per-field granularity (the anti-LWW-record property) ----------
    phone = Operation(new_op_id(), "phone", "entry", "e4", "title", "set", "New title", 5, 10.0)
    laptop = Operation(new_op_id(), "laptop", "entry", "e4", "notes", "set", "New notes", 5, 5.0)
    merged_state = resolve([phone, laptop])
    check(
        "Field granularity: concurrent edits to different fields both survive",
        merged_state["e4"]["title"] == "New title" and merged_state["e4"]["notes"] == "New notes",
    )

    # -- 5. ORSet: re-add after remove, no resurrection --------------------
    rA = ReplicaState("A")
    add1 = rA.make_op("entry", "e1", "tags", "add", "urgent")
    rem1 = rA.make_remove_op("entry", "e1", "tags", "urgent")
    check("ORSet: remove hides the element", rA.snapshot()["e1"]["tags"] == [])
    rA.apply(add1)  # duplicate delivery of the SAME add
    check("ORSet: replayed old add does NOT resurrect", rA.snapshot()["e1"]["tags"] == [])
    dup_add = Operation(add1.op_id, add1.replica_id, add1.entity, add1.entity_id,
                        add1.field, add1.action, add1.value, add1.lamport + 50, 9e9)
    rA.apply(dup_add)  # same tag arriving with a forged, later stamp
    check("ORSet: same add-tag re-delivered with a forged clock does NOT resurrect",
          rA.snapshot()["e1"]["tags"] == [])
    add2 = rA.make_op("entry", "e1", "tags", "add", "urgent")  # genuinely NEW add
    check("ORSet: a genuinely new add after remove DOES restore", rA.snapshot()["e1"]["tags"] == ["urgent"])
    orders = []
    for perm in ([add1, rem1, dup_add, add2], [add2, dup_add, rem1, add1], [rem1, add2, add1, dup_add]):
        r = ReplicaState("t")
        r.apply_many(list(perm))
        orders.append(canon(r.snapshot()["e1"]["tags"]))
    check("ORSet: re-add outcome independent of arrival order", len(set(orders)) == 1 and orders[0] == canon(["urgent"]))

    # bare-element (tag-less) remove falls back to the causal threshold
    rB = ReplicaState("B")
    bare_add = rB.make_op("entry", "e2", "tags", "add", "draft")
    rB.make_op("entry", "e2", "tags", "remove", "draft")  # legacy client: no tags
    check("ORSet: tag-less remove still removes", rB.snapshot()["e2"]["tags"] == [])
    rB.apply(bare_add)
    check("ORSet: tag-less remove is not undone by replay", rB.snapshot()["e2"]["tags"] == [])
    rB.make_op("entry", "e2", "tags", "add", "draft")
    check("ORSet: new add after tag-less remove works", rB.snapshot()["e2"]["tags"] == ["draft"])

    # -- 6. GSet -----------------------------------------------------------
    g1, g2 = GSet(), GSet()
    g1.apply(Operation(new_op_id(), "r1", "entry", "e1", "hist", "add", "opened", 1, 0.0))
    g2.apply(Operation(new_op_id(), "r2", "entry", "e1", "hist", "add", "shared", 1, 0.0))
    check("GSet merge is union & symmetric", canon(g1.merge(g2).value()) == canon(g2.merge(g1).value()))
    check("GSet: removes are ignored", len(g1.merge(g2).value()) == 2)

    # -- 7. Delete tombstone propagation ----------------------------------
    d1, d2 = ReplicaState("d1"), ReplicaState("d2")
    seed = d1.make_op("entry", "e7", "title", "set", "Doomed")
    d2.apply(seed)
    del_op = d1.make_op("entry", "e7", "", "delete", None)
    edit_op = d2.make_op("entry", "e7", "title", "set", "Renamed offline")  # concurrent
    d1.apply(edit_op)
    d2.apply(del_op)
    check("Delete: tombstone survives a concurrent edit on both replicas",
          d1.snapshot()["e7"]["_deleted"] and d2.snapshot()["e7"]["_deleted"])
    check("Delete: replicas still converge", canon(d1.snapshot()) == canon(d2.snapshot()))
    check("Delete: entity hidden from live view", "e7" not in d1.live_snapshot())
    d3 = ReplicaState("d3")
    d3.apply_many([seed])  # a replica that never heard about the delete
    d3.merge(d1)
    check("Delete: merging a tombstone propagates it", d3.snapshot()["e7"]["_deleted"])
    late = Operation(new_op_id(), "d9", "entry", "e7", "title", "set", "zombie", 9999, 9e9)
    d3.apply(late)
    check("Delete: a late edit cannot resurrect a deleted entity", d3.snapshot()["e7"]["_deleted"])

    # -- 8. diff_for_sync / ops_since -------------------------------------
    server = ReplicaState("server")
    client = ReplicaState("client")
    server_ops: List[Operation] = []
    for i in range(12):
        server_ops.append(server.make_op("entry", f"s{i % 3}", "title", "set", f"v{i}"))
    for i, o in enumerate(server_ops):
        if i < 7:
            client.apply(o)
    expected = {o.op_id for o in server_ops[7:]}
    diff = diff_for_sync(client.clock.to_dict(), server.ops())
    check("diff_for_sync returns exactly the missing ops", {o.op_id for o in diff} == expected)
    check("diff_for_sync is in causal order", [o.sort_key() for o in diff] == sorted(o.sort_key() for o in diff))
    check("ops_since agrees with diff_for_sync", {o.op_id for o in server.ops_since(client.clock)} == expected)
    client.apply_many(diff)
    check("After applying the diff the client converges", canon(client.snapshot()) == canon(server.snapshot()))
    check("Empty clock asks for everything", len(diff_for_sync({}, server.ops())) == len(server_ops))
    check("Up-to-date clock asks for nothing", diff_for_sync(server.clock.to_dict(), server.ops()) == [])
    check("Duplicate diff application is a no-op", client.apply_many(diff) == 0)

    # -- 9. merge_operations ----------------------------------------------
    m1 = merge_operations(server_ops[:8], server_ops[5:])
    m2 = merge_operations(server_ops[5:], server_ops[:8])
    check("merge_operations dedupes and is commutative",
          [o.op_id for o in m1] == [o.op_id for o in m2] == [o.op_id for o in sorted(server_ops, key=Operation.sort_key)])

    # -- 10. Serialisation round-trips ------------------------------------
    restored = ReplicaState.from_dict(json.loads(json.dumps(server.to_dict())))
    check("ReplicaState round-trips through JSON", canon(restored.snapshot()) == canon(server.snapshot()))
    check("ReplicaState round-trip keeps the op log", len(restored) == len(server))
    orset_rt = ORSet.from_dict(json.loads(json.dumps(rA._entities["e1"].orsets["tags"].to_dict())))
    check("ORSet round-trips (tags, tombstones, bound)", canon(orset_rt.value()) == canon(["urgent"]))
    lww_rt = LWWRegister.from_dict(json.loads(json.dumps(sa._entities["e9"].registers["title"].to_dict())))
    check("LWWRegister round-trips with its key", lww_rt.value() == "causal")

    # -- 11. GSet via schema ----------------------------------------------
    sch = ReplicaState("s", {"entry.history": KIND_GSET})
    h1 = sch.make_op("entry", "e1", "history", "add", "read")
    sch.make_op("entry", "e1", "history", "remove", "read")
    check("Schema: a GSet field ignores removes", sch.snapshot()["e1"]["history"] == ["read"])

    # -- 12. Randomised convergence property test -------------------------
    ENTITIES = [f"e{i}" for i in range(6)]
    FIELDS = ["title", "summary", "body", "rating"]
    TAGS = ["urgent", "work", "read-later", "archive", "idea"]

    def build_history(rng: random.Random, n_replicas: int, n_ops: int) -> List[Operation]:
        """Simulate replicas editing offline with occasional partial syncs.

        Each replica applies its own ops immediately (so its removes are real
        observed-removes) and now and then receives a random slice of a peer's
        log -- which is what creates genuine causality *and* genuine concurrency.
        """
        reps = [ReplicaState(f"rep-{i}") for i in range(n_replicas)]
        log: List[Operation] = []
        for _ in range(n_ops):
            r = rng.choice(reps)
            eid = rng.choice(ENTITIES)
            roll = rng.random()
            if roll < 0.40:
                o = r.make_op("entry", eid, rng.choice(FIELDS), "set", rng.randint(0, 999))
            elif roll < 0.65:
                o = r.make_op("entry", eid, "tags", "add", rng.choice(TAGS))
            elif roll < 0.85:
                tag = rng.choice(TAGS)
                if rng.random() < 0.75:
                    o = r.make_remove_op("entry", eid, "tags", tag)
                else:  # legacy client: bare element, no observed tags
                    o = r.make_op("entry", eid, "tags", "remove", tag)
            elif roll < 0.90:
                o = r.make_op("entry", eid, "", "delete", None)
            else:
                o = r.make_op("note", eid, "text", "set", f"n{rng.randint(0, 99)}")
            log.append(o)
            if rng.random() < 0.25:  # partial, lossy gossip between devices
                src, dst = rng.choice(reps), rng.choice(reps)
                if src is not dst and len(src):
                    batch = src.ops()
                    dst.apply_many(batch[: rng.randint(1, len(batch))])
        return log

    SEEDS = 200
    OPS_PER_SEED = 400
    TRIALS = 4
    total_ops = 0
    conv_ok = True
    resolve_ok = True
    assoc_ok = True
    detail = ""
    for seed in range(SEEDS):
        rng = random.Random(seed)
        n_replicas = rng.choice([3, 4, 5])
        history = build_history(rng, n_replicas, OPS_PER_SEED)
        total_ops += len(history)

        snapshots = []
        for t in range(TRIALS):
            trng = random.Random(seed * 1000 + t)
            delivery = list(history)
            # duplicates: at-least-once delivery, retried queues, replays
            delivery += [trng.choice(history) for _ in range(len(history) // 4)]
            trng.shuffle(delivery)
            rep = ReplicaState(f"conv-{t}")
            # deliver in ragged chunks, interleaved, to mimic real sync batches
            i = 0
            while i < len(delivery):
                step = trng.randint(1, 25)
                rep.apply_many(delivery[i : i + step])
                i += step
            snapshots.append(canon(rep.snapshot()))

        if len(set(snapshots)) != 1:
            conv_ok = False
            detail = f"seed {seed}: replicas diverged"
            break
        if canon(resolve(list(reversed(history)))) != snapshots[0]:
            resolve_ok = False
            detail = f"seed {seed}: resolve() disagreed with replica state"
            break

        # associativity: merge three disjoint partial replicas both ways
        rng.shuffle(history)
        third = max(1, len(history) // 3)
        parts = [history[:third], history[third : 2 * third], history[2 * third :]]
        left = ReplicaState("L")
        left.apply_many(parts[0])
        mid = ReplicaState("M")
        mid.apply_many(parts[1])
        right = ReplicaState("R")
        right.apply_many(parts[2])
        lm = ReplicaState("LM")
        lm.merge(left)
        lm.merge(mid)
        lm.merge(right)
        mr = ReplicaState("MR")
        mr.merge(mid)
        mr.merge(right)
        rl = ReplicaState("RL")
        rl.merge(mr)
        rl.merge(left)
        rl.merge(rl)  # merge with self: idempotent
        if canon(lm.snapshot()) != canon(rl.snapshot()) != snapshots[0]:
            assoc_ok = False
            detail = f"seed {seed}: merge is not associative"
            break

    check(f"Randomised convergence: {SEEDS} seeds x ~{OPS_PER_SEED} ops x {TRIALS} shuffled/duplicated deliveries", conv_ok, detail)
    check("resolve() matches replica state for every seed", resolve_ok, detail)
    check("ReplicaState.merge is associative + idempotent over random histories", assoc_ok, detail)

    print(f"\n{total_ops} operations exercised across {SEEDS} seeds.")
    if _FAILURES:
        print(f"RESULT: FAIL ({len(_FAILURES)} failing): " + ", ".join(_FAILURES))
        raise SystemExit(1)
    print("RESULT: ALL PROPERTIES PASS")
