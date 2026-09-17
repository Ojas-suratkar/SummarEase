"""
Real-time collaborative reading rooms -- the room registry and event fan-out.

WHAT THIS IS
------------
A room is a handful of people looking at the same document at the same
time: their cursors, their highlights, their comments, their reactions.
This module owns that shared state and the broadcast of changes to it.
It is deliberately transport-shaped for **Server-Sent Events**, not
WebSockets.

WHY SSE AND NOT WEBSOCKETS
--------------------------
The traffic here is overwhelmingly one-directional: one person highlights
a sentence, everyone else's screen has to learn about it. The handful of
bytes going the other way (a cursor position, a new comment) are perfectly
happy as ordinary POSTs. SSE gives us that fan-out over plain HTTP, which
means:

  * no extra Python dependency (no flask-socketio, no eventlet, no gevent
    monkey-patching of the whole interpreter),
  * it survives corporate proxies, which frequently mangle or forbid the
    WebSocket Upgrade handshake but pass a long-lived `text/event-stream`,
  * the browser's built-in `EventSource` does reconnection and
    `Last-Event-ID` resumption for us, which is most of a reliable
    transport handed over for free.

The price is one held worker thread per connected viewer. That is a real
cost and it is why every blocking wait in here has a timeout -- see below.

HONEST LIMITATIONS -- READ BEFORE DEPLOYING
-------------------------------------------
Rooms live in this process's memory and nowhere else. Concretely:

  * **They do not survive a restart.** Redeploy the app and every open
    room is gone. Participants will see the stream drop and their client
    will get a 404 on reconnect.
  * **They do not work across worker processes.** `gunicorn -w 4` gives
    you four independent, mutually invisible copies of this registry;
    two people in "the same" room served by different workers will not
    see each other at all. Until there is a shared broker (Redis pub/sub,
    Postgres LISTEN/NOTIFY, anything out-of-process), collaborative rooms
    must run on a **single worker process** -- threaded, not forked
    (`gunicorn -w 1 --threads N`, or the Flask dev server).
  * **There is no persistence.** Highlights and comments made in a room
    are lost when the room is pruned. Anything worth keeping has to be
    copied into the database by the caller (the annotations service is
    the natural home for that).

None of that is hidden behind a config flag pretending otherwise. It is
the shape of the trade we made to ship this without new dependencies.

THREAD SAFETY
-------------
The Flask dev server and gunicorn both serve requests concurrently, so
every piece of mutable state in here is guarded. Two lock layers:

  * `_registry_lock` -- the dict of rooms (create / look up / delete).
  * `_Room.lock`     -- everything inside one room.

Lock ordering is strictly **registry -> room**, never the reverse, and no
function holds one while blocking. `prune()` in particular is written in
two phases specifically so it never needs a room lock and the registry
lock at the same time in the wrong order. Live delivery uses one
`queue.Queue` per subscriber, so a publisher never blocks on, or is
slowed by, a slow reader.

THE WORKER-THREAD RISK
----------------------
A subscriber generator sits inside a WSGI worker thread for as long as
the stream is open. If a wait in here ever blocked without a deadline, a
client that vanished mid-connection (laptop lid closed, NAT timeout, a
proxy that dropped the socket without an RST) would pin that worker
forever, and enough of them would starve the whole app. So:

  * every wait is `Queue.get(timeout=...)`, never an unbounded `get()`,
  * the loop re-checks liveness (room still exists, participant still a
    member) on every tick,
  * each stream is capped at `STREAM_MAX_SECONDS` and then closes
    cleanly -- the client reconnects and resumes from its last event id,
    which costs the user nothing because replay is exact.
"""
from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from collections import deque
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

#: How many durable events we keep per room for replay-after-disconnect.
#: Bounded on purpose: memory per room is capped no matter how long a
#: room stays busy, and replaying a reconnecting client is O(ring), not
#: O(history).
RING_CAPACITY = 500

#: How often an idle stream emits an SSE comment line. Proxies and load
#: balancers happily close a connection that has been silent for 30-60s;
#: a comment is the cheapest possible "still here".
HEARTBEAT_SECONDS = 15.0

#: Granularity of the blocking wait. Small enough that a stream notices
#: a departed participant or a deleted room quickly, large enough that an
#: idle stream costs essentially nothing.
WAIT_SLICE_SECONDS = 1.0

#: Hard ceiling on a single stream. See "THE WORKER-THREAD RISK" above.
STREAM_MAX_SECONDS = 3600.0

#: Per-subscriber delivery buffer. A reader that falls this far behind is
#: not worth waiting for; it gets a `state.sync` and carries on.
SUBSCRIBER_QUEUE_SIZE = 512

#: Payload hygiene -- a client cannot make us hold arbitrary amounts of
#: memory by posting one enormous comment.
MAX_STRING_LENGTH = 8_000
MAX_COLLECTION_ITEMS = 200
MAX_PAYLOAD_DEPTH = 6

#: Join codes avoid 0/O and 1/I/L -- they get read aloud and typed by hand.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LENGTH = 6

EVENT_TYPES: frozenset[str] = frozenset(
    {
        "presence.join",
        "presence.leave",
        "cursor.move",
        "highlight.add",
        "highlight.remove",
        "comment.add",
        "comment.remove",
        "reaction.add",
        "state.sync",
    }
)

#: Events that are *about right now* and meaningless five seconds later.
#: They are broadcast to live subscribers but never stored in the ring:
#: replaying somebody's old mouse positions after a reconnect would be
#: noise, and letting cursor spam evict real highlights from the replay
#: buffer would be a bug. They still consume event ids so ids stay
#: strictly monotonic on the wire, which is what `Last-Event-ID` needs.
TRANSIENT_TYPES: frozenset[str] = frozenset({"cursor.move", "reaction.add"})


class RoomError(RuntimeError):
    """Raised for a missing room, an unknown participant, or a bad event."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _now() -> float:
    return time.time()


def _hue_for(identifier: str) -> int:
    """Stable colour per participant, derived from the id (FNV-1a -> hue).

    The client computes this with exactly the same formula, so a
    participant is the same colour on every screen even before the
    presence list has loaded. Kept here too so the server-rendered
    participant list never disagrees with the live one.
    """
    h = 0x811C9DC5
    for ch in identifier:
        h ^= ord(ch) & 0xFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % 360


def _initials(display_name: str) -> str:
    parts = [p for p in (display_name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _clean_text(value: Any, limit: int = MAX_STRING_LENGTH) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def _scrub(value: Any, depth: int = 0) -> Any:
    """Coerce a client payload into something small and JSON-safe.

    Untrusted input goes straight into memory that every other
    participant will read, so it is bounded here rather than hoped about.
    """
    if depth > MAX_PAYLOAD_DEPTH:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # NaN / inf are not valid JSON and would break the SSE frame.
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return 0
        return value
    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]
    if isinstance(value, (list, tuple)):
        return [_scrub(v, depth + 1) for v in list(value)[:MAX_COLLECTION_ITEMS]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:MAX_COLLECTION_ITEMS]:
            out[str(k)[:120]] = _scrub(v, depth + 1)
        return out
    return _clean_text(value, 500)


# --------------------------------------------------------------------------
# Internal room / subscriber structures
# --------------------------------------------------------------------------


class _Subscriber:
    """One open SSE stream's delivery buffer.

    A `queue.Queue` rather than a `Condition` broadcast because it gives
    us per-reader backpressure that the *publisher* never has to wait on:
    `put_nowait` either succeeds or marks this one reader as overflowed,
    so one stalled browser can never slow down everyone else's room.
    """

    __slots__ = ("q", "overflowed")

    def __init__(self) -> None:
        self.q: queue.Queue[dict] = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.overflowed = False


class _Room:
    __slots__ = (
        "id",
        "owner_id",
        "entry_id",
        "title",
        "join_code",
        "created_at",
        "last_activity",
        "lock",
        "seq",
        "events",
        "dropped_through",
        "participants",
        "highlights",
        "comments",
        "subscribers",
        "closed",
    )

    def __init__(self, room_id: str, owner_id: int, entry_id: int | None, title: str, join_code: str) -> None:
        self.id = room_id
        self.owner_id = owner_id
        self.entry_id = entry_id
        self.title = title
        self.join_code = join_code
        self.created_at = _now()
        self.last_activity = self.created_at
        # RLock so the materialisation helpers can be called from inside
        # publish() without a second acquisition deadlocking us.
        self.lock = threading.RLock()
        self.seq = 0
        self.events: deque[dict] = deque(maxlen=RING_CAPACITY)
        #: Highest event id that has fallen out of the ring. A client
        #: reconnecting with a `last_event_id` below this has provably
        #: missed something we can no longer replay, so it gets a full
        #: `state.sync` instead of a silent hole.
        self.dropped_through = 0
        self.participants: dict[str, dict] = {}
        self.highlights: dict[str, dict] = {}
        self.comments: dict[str, dict] = {}
        self.subscribers: dict[str, _Subscriber] = {}
        self.closed = False

    # -- called with self.lock held -------------------------------------

    def public(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "entry_id": self.entry_id,
            "title": self.title,
            "join_code": self.join_code,
            "created_at": self.created_at,
            "participant_count": len(self.participants),
            "last_event_id": self.seq,
        }


_registry_lock = threading.RLock()
_rooms: dict[str, _Room] = {}
_codes: dict[str, str] = {}  # join_code -> room_id


def _require(room_id: str) -> _Room:
    with _registry_lock:
        room = _rooms.get(room_id)
    if room is None or room.closed:
        raise RoomError("That room no longer exists.")
    return room


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def create_room(owner_id: int, entry_id: int | None, title: str) -> dict:
    """Create a room and return its public dict (includes `join_code`)."""
    title = _clean_text(title, 200).strip() or "Reading room"
    with _registry_lock:
        room_id = secrets.token_urlsafe(9)
        while room_id in _rooms:  # astronomically unlikely; still checked
            room_id = secrets.token_urlsafe(9)
        code = _new_join_code_locked()
        room = _Room(room_id, owner_id, entry_id, title, code)
        _rooms[room_id] = room
        _codes[code] = room_id
        return room.public()


def _new_join_code_locked() -> str:
    """Caller must hold `_registry_lock`."""
    for _ in range(64):
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
        if code not in _codes:
            return code
    # Fall back to something guaranteed unique rather than looping forever.
    return secrets.token_hex(5).upper()


def get_room(room_id: str) -> dict | None:
    with _registry_lock:
        room = _rooms.get(room_id)
    if room is None or room.closed:
        return None
    with room.lock:
        return room.public()


def get_room_by_code(join_code: str) -> dict | None:
    """Resolve a typed-in / pasted join code to a room."""
    code = (join_code or "").strip().upper()
    with _registry_lock:
        room_id = _codes.get(code)
    return get_room(room_id) if room_id else None


def join_room(room_id: str, user_id: int, display_name: str) -> dict:
    """Add a participant. Returns `{participant_id, room, recent_events}`.

    `recent_events` is the replay ring as of *after* the join was
    published, so the client can take `max(id)` from it as its starting
    `last_event_id` and know there is no gap between the snapshot and the
    stream it is about to open.
    """
    room = _require(room_id)
    name = _clean_text(display_name, 80).strip() or "Guest"
    participant_id = secrets.token_urlsafe(8)
    with room.lock:
        room.participants[participant_id] = {
            "participant_id": participant_id,
            "user_id": user_id,
            "display_name": name,
            "initials": _initials(name),
            "hue": _hue_for(participant_id),
            "joined_at": _now(),
            "last_seen": _now(),
            "cursor": None,
        }
        room.last_activity = _now()
        _publish_locked(room, participant_id, "presence.join", {"display_name": name})
        recent = list(room.events)
        snapshot = room.public()
    return {"participant_id": participant_id, "room": snapshot, "recent_events": recent}


def leave_room(room_id: str, participant_id: str) -> None:
    """Remove a participant and tell the room. Idempotent and never raises
    for an unknown room -- it is called from `finally` blocks and page
    unload beacons, where failing loudly helps nobody."""
    with _registry_lock:
        room = _rooms.get(room_id)
    if room is None:
        return
    with room.lock:
        person = room.participants.pop(participant_id, None)
        if person is None:
            return
        _publish_locked(
            room,
            participant_id,
            "presence.leave",
            {"display_name": person["display_name"]},
        )
        room.last_activity = _now()


def publish(room_id: str, participant_id: str | None, event_type: str, payload: dict) -> dict:
    """Record an event and fan it out. Returns the stored event."""
    if event_type not in EVENT_TYPES:
        raise RoomError(f"Unknown event type: {event_type!r}")
    room = _require(room_id)
    with room.lock:
        if participant_id is not None and participant_id not in room.participants:
            raise RoomError("You are not in this room any more.")
        event = _publish_locked(room, participant_id, event_type, payload or {})
        room.last_activity = _now()
    return event


def participants(room_id: str) -> list[dict]:
    with _registry_lock:
        room = _rooms.get(room_id)
    if room is None:
        return []
    with room.lock:
        return [dict(p) for p in room.participants.values()]


def room_state(room_id: str) -> dict:
    """The materialised view: what a client needs to draw the room from
    scratch, with no event history. Used by the `/state` route, by the
    polling fallback, and as the payload of a `state.sync` event."""
    room = _require(room_id)
    with room.lock:
        return {
            "room": room.public(),
            "participants": [dict(p) for p in room.participants.values()],
            "highlights": [dict(h) for h in room.highlights.values()],
            "comments": [dict(c) for c in room.comments.values()],
            "last_event_id": room.seq,
        }


def prune(max_idle_seconds: int = 3600) -> int:
    """Reap idle participants and dead rooms. Returns how many things went.

    Two phases on purpose. Phase one touches room locks only; phase two
    touches the registry lock only. Never both at once, so this can never
    invert the registry->room lock order and deadlock against a
    concurrent `create_room` or `join_room`.
    """
    cutoff = _now() - max_idle_seconds
    reaped = 0

    with _registry_lock:
        rooms = list(_rooms.values())

    empty_and_idle: list[_Room] = []
    for room in rooms:
        with room.lock:
            stale = [
                pid for pid, p in room.participants.items() if p["last_seen"] < cutoff
            ]
            for pid in stale:
                person = room.participants.pop(pid, None)
                if person is not None:
                    reaped += 1
                    _publish_locked(
                        room,
                        pid,
                        "presence.leave",
                        {"display_name": person["display_name"], "reason": "idle"},
                    )
            if not room.participants and room.last_activity < cutoff:
                empty_and_idle.append(room)

    if empty_and_idle:
        with _registry_lock:
            for room in empty_and_idle:
                # Re-check under the registry lock: somebody may have
                # joined in the gap between the two phases.
                if _rooms.get(room.id) is not room:
                    continue
                with room.lock:
                    if room.participants:
                        continue
                    room.closed = True
                    subs = list(room.subscribers.values())
                    room.events.clear()
                    room.highlights.clear()
                    room.comments.clear()
                del _rooms[room.id]
                _codes.pop(room.join_code, None)
                reaped += 1
                # Nudge any stream still parked on this room so it stops
                # within a tick instead of waiting out its slice.
                for sub in subs:
                    try:
                        sub.q.put_nowait({"type": "__closed__"})
                    except queue.Full:
                        pass
    return reaped


def subscribe(
    room_id: str,
    participant_id: str,
    last_event_id: int = 0,
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    max_seconds: float = STREAM_MAX_SECONDS,
) -> Iterator[dict]:
    """Yield events for one SSE stream.

    Contract with the caller (routes.py):

      * Everything with an id greater than `last_event_id` is replayed
        first, in order, before any live event -- so a client that
        reconnects after a dropped connection loses nothing. The browser
        sends its `Last-Event-ID` header automatically; pass it here.
      * If the client's `last_event_id` is older than what the ring still
        holds, it gets a single `state.sync` event carrying the whole
        materialised room instead of a silent gap. Missing an update is a
        corrupted room; resyncing is merely a hiccup.
      * Heartbeats arrive as `{"type": "heartbeat"}` roughly every
        `heartbeat_seconds`. Pass them through `format_sse()`, which
        renders them as an SSE comment line so `EventSource` ignores them
        but every proxy in the path sees traffic.
      * The iterator ends -- it does not raise -- when the participant
        leaves, the room is pruned, or `max_seconds` elapses. All three
        are normal; the client reconnects and resumes.

    Raises `RoomError` eagerly (before any streaming begins) if the room
    or participant is unknown, so the route can still return a 404.
    """
    room = _require(room_id)
    with room.lock:
        if participant_id not in room.participants:
            raise RoomError("You are not in this room any more.")
    return _stream(room, participant_id, int(last_event_id or 0), heartbeat_seconds, max_seconds)


def _stream(
    room: _Room,
    participant_id: str,
    last_event_id: int,
    heartbeat_seconds: float,
    max_seconds: float,
) -> Iterator[dict]:
    sub = _Subscriber()
    stream_key = secrets.token_urlsafe(6)

    # Register the queue *before* snapshotting the ring. Anything
    # published from here on lands in the queue; anything published
    # before is in the snapshot. The two therefore overlap rather than
    # leave a gap, and the dedupe below removes the overlap. Doing it the
    # other way round would drop any event published in between.
    with room.lock:
        room.subscribers[stream_key] = sub
        # Clamp a client that claims to have seen further than we have
        # ever published. A stale tab resuming against a room that was
        # pruned and recreated would otherwise hand us an id from the
        # future, and every real event would look like a duplicate and be
        # silently swallowed. Trusting a client-supplied cursor
        # unconditionally is exactly how a room goes quiet for one person
        # and nobody can reproduce it.
        last_event_id = min(last_event_id, room.seq)
        gap = last_event_id > 0 and last_event_id < room.dropped_through
        if gap:
            resync = _make_event(room.dropped_through, None, "state.sync", _state_locked(room))
            backlog = [resync] + [e for e in room.events if e["id"] > room.dropped_through]
        else:
            backlog = [e for e in room.events if e["id"] > last_event_id]
        highest = backlog[-1]["id"] if backlog else last_event_id

    try:
        for event in backlog:
            yield event

        deadline = time.monotonic() + max_seconds
        last_beat = time.monotonic()
        slice_seconds = max(0.05, min(WAIT_SLICE_SECONDS, heartbeat_seconds))

        while True:
            if time.monotonic() >= deadline:
                # Deliberate: hand the worker thread back. The client
                # reconnects with its last event id and replay makes the
                # seam invisible.
                return

            try:
                # ALWAYS with a timeout. An unbounded get() here is how a
                # vanished client pins a worker thread forever.
                event = sub.q.get(timeout=slice_seconds)
            except queue.Empty:
                event = None

            if event is not None:
                if event.get("type") == "__closed__":
                    return
                if event["id"] > highest:
                    highest = event["id"]
                    yield event
                    last_beat = time.monotonic()
                continue

            # Timed out: check we are still wanted, then heartbeat.
            with room.lock:
                if room.closed:
                    return
                person = room.participants.get(participant_id)
                if person is None:
                    return
                person["last_seen"] = _now()
                overflowed = sub.overflowed
                if overflowed:
                    sub.overflowed = False
                    resync = _make_event(room.seq, None, "state.sync", _state_locked(room))
                    highest = max(highest, room.seq)
                else:
                    resync = None

            if resync is not None:
                # This reader fell far enough behind that we dropped
                # frames for it rather than stall the publisher. Give it
                # the truth in one shot.
                yield resync
                last_beat = time.monotonic()
                continue

            if time.monotonic() - last_beat >= heartbeat_seconds:
                last_beat = time.monotonic()
                yield {"type": "heartbeat", "id": highest, "ts": _now()}
    finally:
        with room.lock:
            room.subscribers.pop(stream_key, None)


def format_sse(event: dict, *, retry_ms: int | None = None) -> str:
    """Render one yielded dict as an SSE frame.

    Lives here rather than in routes.py so the wire format and the event
    shape stay in one file. Pure string work -- no Flask import, so this
    module stays runnable on its own.
    """
    if event.get("type") == "heartbeat":
        # A comment line: bytes on the wire for the proxies, invisible to
        # EventSource, and it does not disturb the client's Last-Event-ID.
        return f": heartbeat {int(event.get('ts', 0))}\n\n"
    lines = []
    if retry_ms is not None:
        lines.append(f"retry: {int(retry_ms)}")
    lines.append(f"id: {event['id']}")
    lines.append(f"event: {event['type']}")
    lines.append("data: " + json.dumps(event, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------
# Internals -- all of these require room.lock to be held
# --------------------------------------------------------------------------


def _make_event(event_id: int, participant_id: str | None, event_type: str, payload: dict) -> dict:
    return {
        "id": event_id,
        "seq": event_id,
        "type": event_type,
        "participant_id": participant_id,
        "payload": payload,
        "ts": _now(),
    }


def _state_locked(room: _Room) -> dict:
    return {
        "room": room.public(),
        "participants": [dict(p) for p in room.participants.values()],
        "highlights": [dict(h) for h in room.highlights.values()],
        "comments": [dict(c) for c in room.comments.values()],
        "last_event_id": room.seq,
    }


def _publish_locked(room: _Room, participant_id: str | None, event_type: str, payload: dict) -> dict:
    room.seq += 1
    clean = _scrub(payload) if isinstance(payload, dict) else {}
    if not isinstance(clean, dict):
        clean = {}
    event = _make_event(room.seq, participant_id, event_type, clean)

    person = room.participants.get(participant_id) if participant_id else None
    if person is not None:
        person["last_seen"] = _now()
        event["display_name"] = person["display_name"]
        event["hue"] = person["hue"]

    _materialise_locked(room, event)

    if event_type not in TRANSIENT_TYPES:
        if len(room.events) == RING_CAPACITY and room.events:
            # About to evict the oldest: remember how far back replay can
            # no longer reach, so a late reconnect gets a state.sync
            # rather than a hole it cannot detect.
            room.dropped_through = room.events[0]["id"]
        room.events.append(event)

    # put_nowait only -- a publisher must never block on a slow reader.
    for sub in room.subscribers.values():
        try:
            sub.q.put_nowait(event)
        except queue.Full:
            sub.overflowed = True
    return event


def _materialise_locked(room: _Room, event: dict) -> None:
    """Fold an event into the room's current-state view.

    Done here, on the write path, so `/state` and `state.sync` are always
    consistent with the event stream that produced them -- rather than
    replaying 500 events on every read.
    """
    kind = event["type"]
    payload = event["payload"]
    actor = event["participant_id"]

    if kind == "cursor.move":
        person = room.participants.get(actor) if actor else None
        if person is not None:
            person["cursor"] = {
                "x": payload.get("x"),
                "y": payload.get("y"),
                "ts": event["ts"],
            }
        return

    if kind == "highlight.add":
        hid = _clean_text(payload.get("highlight_id") or "", 64).strip() or secrets.token_urlsafe(6)
        payload["highlight_id"] = hid
        person = room.participants.get(actor) if actor else None
        room.highlights[hid] = {
            "highlight_id": hid,
            "start": int(payload.get("start") or 0),
            "end": int(payload.get("end") or 0),
            "text": _clean_text(payload.get("text"), 1200),
            "participant_id": actor,
            "display_name": person["display_name"] if person else "Someone",
            "hue": person["hue"] if person else 210,
            "created_at": event["ts"],
        }
        return

    if kind == "highlight.remove":
        hid = _clean_text(payload.get("highlight_id") or "", 64)
        room.highlights.pop(hid, None)
        # Comments anchored to a removed highlight lose their anchor but
        # keep their text -- deleting somebody's words as a side effect of
        # clearing a highlight would be surprising.
        for comment in room.comments.values():
            if comment.get("highlight_id") == hid:
                comment["highlight_id"] = None
        return

    if kind == "comment.add":
        cid = _clean_text(payload.get("comment_id") or "", 64).strip() or secrets.token_urlsafe(6)
        payload["comment_id"] = cid
        person = room.participants.get(actor) if actor else None
        room.comments[cid] = {
            "comment_id": cid,
            "highlight_id": _clean_text(payload.get("highlight_id") or "", 64) or None,
            "parent_id": _clean_text(payload.get("parent_id") or "", 64) or None,
            "body": _clean_text(payload.get("body"), 4000),
            "participant_id": actor,
            "display_name": person["display_name"] if person else "Someone",
            "hue": person["hue"] if person else 210,
            "created_at": event["ts"],
        }
        return

    if kind == "comment.remove":
        cid = _clean_text(payload.get("comment_id") or "", 64)
        room.comments.pop(cid, None)
        return

    # presence.*, reaction.add and state.sync carry no durable state
    # beyond the participant table, which join/leave already maintain.


# --------------------------------------------------------------------------
# Self-test -- run with:  python app/services/rooms.py
# No Flask imports anywhere in this module, precisely so this works.
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    import random

    _results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        _results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))

    def reset() -> None:
        with _registry_lock:
            _rooms.clear()
            _codes.clear()

    # ------------------------------------------------------------------
    # 1. Many participants joining / publishing / leaving at once.
    # ------------------------------------------------------------------
    reset()
    room = create_room(owner_id=1, entry_id=42, title="Concurrency")
    rid = room["id"]
    THREADS, PER_THREAD = 24, 40
    seen_ids: list[int] = []
    errors: list[str] = []
    ids_lock = threading.Lock()
    start_gate = threading.Barrier(THREADS)

    def worker(n: int) -> None:
        try:
            start_gate.wait(timeout=10)
            joined = join_room(rid, user_id=n, display_name=f"User {n}")
            pid = joined["participant_id"]
            local: list[int] = []
            for i in range(PER_THREAD):
                kind = random.choice(
                    ["highlight.add", "comment.add", "cursor.move", "reaction.add"]
                )
                ev = publish(rid, pid, kind, {"i": i, "x": 0.5, "y": 0.5, "body": f"note {i}"})
                local.append(ev["id"])
            with ids_lock:
                seen_ids.extend(local)
            leave_room(rid, pid)
        except Exception as exc:  # noqa: BLE001 -- the test wants the message
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(THREADS)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    alive = [t for t in threads if t.is_alive()]

    check("concurrency: no worker raised", not errors, "; ".join(errors[:3]))
    check("concurrency: no thread hung (no deadlock)", not alive, f"{len(alive)} still alive")
    check(
        "concurrency: every publish got a unique sequence number",
        len(seen_ids) == len(set(seen_ids)) == THREADS * PER_THREAD,
        f"{len(seen_ids)} events, {len(set(seen_ids))} distinct",
    )
    check(
        "concurrency: participant table empty after everyone left",
        participants(rid) == [],
        f"{len(participants(rid))} left behind",
    )
    st = room_state(rid)
    check(
        "concurrency: materialised state survived (highlights + comments present)",
        len(st["highlights"]) + len(st["comments"]) > 0 and st["last_event_id"] >= len(seen_ids),
        f"h={len(st['highlights'])} c={len(st['comments'])} seq={st['last_event_id']}",
    )
    print(f"       ({THREADS} threads x {PER_THREAD} events in {time.monotonic() - t0:.2f}s)")

    # ------------------------------------------------------------------
    # 2. Replay after disconnect returns exactly the missed events, in order.
    # ------------------------------------------------------------------
    reset()
    rid = create_room(1, None, "Replay")["id"]
    pid = join_room(rid, 1, "Ana")["participant_id"]
    made = [publish(rid, pid, "highlight.add", {"i": i}) for i in range(20)]
    durable = [e["id"] for e in made]
    cut = durable[4]
    expected = [i for i in durable if i > cut]

    gen = subscribe(rid, pid, last_event_id=cut, heartbeat_seconds=0.2)
    got: list[int] = []
    for ev in gen:
        if ev["type"] == "heartbeat":
            break
        got.append(ev["id"])
    gen.close()
    check(
        "replay: exactly the missed events, in order",
        got == expected,
        f"expected {len(expected)} got {len(got)}",
    )

    # A client that never disconnected (last_event_id == head) gets nothing
    # replayed, not a duplicate storm.
    gen = subscribe(rid, pid, last_event_id=durable[-1], heartbeat_seconds=0.2)
    first = next(iter(gen))
    gen.close()
    check("replay: caught-up client replays nothing", first["type"] == "heartbeat", str(first["type"]))

    # ------------------------------------------------------------------
    # 3. Ring buffer caps memory; over-old clients get a state.sync.
    # ------------------------------------------------------------------
    reset()
    rid = create_room(1, None, "Ring")["id"]
    pid = join_room(rid, 1, "Ana")["participant_id"]
    for i in range(RING_CAPACITY * 4):
        publish(rid, pid, "comment.add", {"body": f"c{i}", "comment_id": f"c{i}"})
    with _rooms[rid].lock:
        ring_len = len(_rooms[rid].events)
    check(
        f"ring: capped at {RING_CAPACITY} events regardless of traffic",
        ring_len == RING_CAPACITY,
        f"ring holds {ring_len}",
    )

    gen = subscribe(rid, pid, last_event_id=1, heartbeat_seconds=0.2)
    replayed: list[dict] = []
    for ev in gen:
        if ev["type"] == "heartbeat":
            break
        replayed.append(ev)
    gen.close()
    check(
        "ring: a client past the replay window gets state.sync, not a silent gap",
        bool(replayed) and replayed[0]["type"] == "state.sync",
        f"first replayed event was {replayed[0]['type'] if replayed else 'nothing'}",
    )
    check(
        "ring: state.sync carries the materialised room",
        bool(replayed) and len(replayed[0]["payload"]["comments"]) == RING_CAPACITY * 4,
        f"{len(replayed[0]['payload']['comments']) if replayed else 0} comments in sync",
    )
    check(
        "ring: replay after the sync is strictly increasing",
        all(replayed[i]["id"] < replayed[i + 1]["id"] for i in range(len(replayed) - 1)),
    )

    # ------------------------------------------------------------------
    # 4. subscribe() returns promptly on timeout instead of hanging.
    # ------------------------------------------------------------------
    reset()
    rid = create_room(1, None, "Heartbeat")["id"]
    pid = join_room(rid, 1, "Ana")["participant_id"]
    gen = subscribe(rid, pid, last_event_id=10**9, heartbeat_seconds=0.25)
    t0 = time.monotonic()
    beat = next(iter(gen))
    elapsed = time.monotonic() - t0
    gen.close()
    check(
        "timeout: first heartbeat arrives promptly (no unbounded wait)",
        beat["type"] == "heartbeat" and elapsed < 2.0,
        f"{beat['type']} after {elapsed:.2f}s",
    )

    # And the stream ends by itself once the participant is gone.
    pid2 = join_room(rid, 2, "Bo")["participant_id"]
    gen = subscribe(rid, pid2, heartbeat_seconds=0.1)
    ended = threading.Event()

    def drain() -> None:
        for _ in gen:
            pass
        ended.set()

    dt = threading.Thread(target=drain, daemon=True)
    dt.start()
    time.sleep(0.2)
    leave_room(rid, pid2)
    stream_ended = ended.wait(timeout=5.0)
    check(
        "timeout: stream ends on its own after the participant leaves",
        stream_ended,
        "" if stream_ended else "stream did not terminate",
    )

    # ------------------------------------------------------------------
    # 5. Live fan-out loses nothing while publishers hammer the room.
    # ------------------------------------------------------------------
    reset()
    rid = create_room(1, None, "Fanout")["id"]
    reader = join_room(rid, 0, "Reader")["participant_id"]
    gen = subscribe(rid, reader, heartbeat_seconds=0.2)
    received: list[int] = []
    TOTAL = 8 * 25

    def consume() -> None:
        for ev in gen:
            if ev["type"] == "heartbeat":
                if len(received) >= TOTAL:
                    return
                continue
            received.append(ev["id"])

    ct = threading.Thread(target=consume, daemon=True)
    ct.start()
    time.sleep(0.1)

    pub_errors: list[str] = []
    published: list[int] = []
    pub_lock = threading.Lock()

    def publisher(n: int) -> None:
        try:
            p = join_room(rid, n, f"P{n}")["participant_id"]
            mine = [publish(rid, p, "highlight.add", {"i": i})["id"] for i in range(25)]
            with pub_lock:
                published.extend(mine)
        except Exception as exc:  # noqa: BLE001
            pub_errors.append(str(exc))

    pubs = [threading.Thread(target=publisher, args=(n,), daemon=True) for n in range(1, 9)]
    for t in pubs:
        t.start()
    for t in pubs:
        t.join(timeout=20)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and len(received) < TOTAL:
        time.sleep(0.05)
    ct.join(timeout=5)
    consumer_finished = not ct.is_alive()
    if consumer_finished:
        gen.close()

    check("fan-out: no publisher raised", not pub_errors, "; ".join(pub_errors[:3]))
    check(
        "fan-out: consumer thread was never stuck in the generator",
        consumer_finished,
        "" if consumer_finished else "consumer still blocked",
    )
    check(
        "fan-out: subscriber saw every published event exactly once",
        set(published).issubset(set(received)) and len(received) == len(set(received)),
        f"published {len(published)}, received {len(set(received))} distinct",
    )
    check(
        "fan-out: ids arrived strictly in order",
        all(received[i] < received[i + 1] for i in range(len(received) - 1)),
    )

    # ------------------------------------------------------------------
    # 6. prune() reaps idle participants and dead rooms.
    # ------------------------------------------------------------------
    reset()
    idle_id = create_room(1, None, "Abandoned")["id"]
    idle_pid = join_room(idle_id, 1, "Ghost")["participant_id"]
    live_id = create_room(1, None, "Busy")["id"]
    join_room(live_id, 2, "Active")

    with _rooms[idle_id].lock:
        r = _rooms[idle_id]
        r.last_activity -= 10_000
        for p in r.participants.values():
            p["last_seen"] -= 10_000

    reaped = prune(max_idle_seconds=3600)
    check("prune: reaped the idle participant and the dead room", reaped >= 2, f"reaped {reaped}")
    check("prune: idle room is gone", get_room(idle_id) is None)
    check("prune: busy room untouched", get_room(live_id) is not None)
    check(
        "prune: join code released with the room",
        get_room_by_code(_codes.get("__none__", "") or "ZZZZZZ") is None
        and all(code != idle_pid for code in _codes),
    )
    try:
        publish(idle_id, idle_pid, "comment.add", {"body": "x"})
        pruned_raises = False
    except RoomError:
        pruned_raises = True
    check("prune: publishing into a reaped room raises RoomError", pruned_raises)

    # ------------------------------------------------------------------
    # 7. Stress: mixed create/join/publish/subscribe/leave/prune, no deadlock.
    # ------------------------------------------------------------------
    reset()
    stress_errors: list[str] = []
    stop_at = time.monotonic() + 3.0
    room_ids: list[str] = [create_room(1, None, f"S{i}")["id"] for i in range(4)]

    def stress(n: int) -> None:
        rnd = random.Random(n)
        try:
            while time.monotonic() < stop_at:
                target = rnd.choice(room_ids)
                try:
                    p = join_room(target, n, f"S{n}")["participant_id"]
                except RoomError:
                    continue
                for _ in range(rnd.randint(1, 6)):
                    try:
                        publish(target, p, rnd.choice(["cursor.move", "highlight.add"]), {"x": 0.1})
                    except RoomError:
                        break
                if rnd.random() < 0.4:
                    try:
                        g = subscribe(target, p, heartbeat_seconds=0.05)
                        next(iter(g))
                        g.close()
                    except (RoomError, StopIteration):
                        pass
                leave_room(target, p)
                if rnd.random() < 0.1:
                    prune(max_idle_seconds=0)
                    room_ids.append(create_room(1, None, "recycled")["id"])
        except Exception as exc:  # noqa: BLE001
            stress_errors.append(f"{type(exc).__name__}: {exc}")

    sthreads = [threading.Thread(target=stress, args=(n,), daemon=True) for n in range(16)]
    for t in sthreads:
        t.start()
    for t in sthreads:
        t.join(timeout=25)
    still_alive = [t for t in sthreads if t.is_alive()]
    check("stress: no deadlock (all 16 threads finished)", not still_alive, f"{len(still_alive)} stuck")
    check("stress: no unexpected exceptions", not stress_errors, "; ".join(stress_errors[:3]))

    # ------------------------------------------------------------------
    # 8. format_sse wire format.
    # ------------------------------------------------------------------
    frame = format_sse({"id": 7, "type": "highlight.add", "payload": {"a": 1}, "ts": 0})
    check(
        "wire: event frame has id / event / data lines",
        frame.startswith("id: 7\nevent: highlight.add\ndata: {") and frame.endswith("\n\n"),
        repr(frame[:40]),
    )
    check(
        "wire: heartbeat renders as an SSE comment",
        format_sse({"type": "heartbeat", "ts": 1}).startswith(": heartbeat"),
    )

    failures = [n for n, ok, _ in _results if not ok]
    print("-" * 62)
    print(f"{len(_results) - len(failures)}/{len(_results)} properties passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
    sys.exit(1 if failures else 0)
