"""
SQLAlchemy models -- the entire persistence layer for the app in one
place, replacing what used to be eight separate hand-rolled sqlite3
modules (each with its own connection, its own `CREATE TABLE IF NOT
EXISTS`, its own `?`-placeholder queries that only work against SQLite).

Two things this fixes on top of being one coherent schema:

1. Every table now carries a `user_id` -- with real accounts, nothing
   here is global anymore. Every query a service module makes is scoped
   to `user_id` from here on, so one account can never see another's data.
2. `HistoryEntry.source_text` actually stores the *full* original text,
   not a 500-character excerpt. Previously the only durable copy of a
   summarized document was that excerpt, and the full text lived only in
   an in-memory, TTL'd cache (`rag.py`'s old `_docs` dict) -- which is
   exactly why "Ask this document", the Credibility Lens, Faithfulness
   Check, flashcard generation, and Perspectives would all start saying
   "this document is no longer available" after a restart or ~2 hours.
   `doc_id` is now just `str(HistoryEntry.id)` -- one identity for "this
   thing you summarized", not two disconnected ones -- and RAG chunks
   (`RagChunk`) are a persisted, lazily-built cache keyed off that same
   id, rebuilt on demand from `source_text` if they're ever missing
   rather than silently expiring.

Works against SQLite locally (the default, `instance/app.db`) and
Postgres in production (`DATABASE_URL`, see config.py) -- the whole
reason this is SQLAlchemy now instead of raw sqlite3.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(120), nullable=False, default="")
    password_hash = db.Column(db.String(255), nullable=False)
    api_token = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password or "")

    @staticmethod
    def new_api_token() -> str:
        return secrets.token_urlsafe(32)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User {self.email}>"


class HistoryEntry(db.Model):
    __tablename__ = "history_entries"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    source_type = db.Column(db.String(20), nullable=False)
    source_ref = db.Column(db.Text)
    summary = db.Column(db.Text, nullable=False)
    source_text = db.Column(db.Text)  # full original text -- durable (see module docstring)
    embedding = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)

    # --- Everything below is what turns a read-only log into something
    # --- you can actually manage: rename it, annotate it, file it away,
    # --- re-run it. Added in the "make this a real product" pass.
    title = db.Column(db.String(300))  # user-editable; falls back to source_ref
    notes = db.Column(db.Text)  # freeform notes the user writes on the entry
    tags = db.Column(db.String(500), default="")  # comma-separated, lowercased
    detail_level = db.Column(db.String(20), default="standard")  # so a re-run can repeat or change it
    is_archived = db.Column(db.Boolean, nullable=False, default=False)
    is_pinned = db.Column(db.Boolean, nullable=False, default=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=_now, onupdate=_now)

    user = db.relationship("User", backref=db.backref("history_entries", lazy="dynamic"))

    @property
    def excerpt(self) -> str:
        return (self.source_text or "")[:500]

    @property
    def display_title(self) -> str:
        """What to show as this entry's name. The user's own title wins;
        otherwise fall back to the source reference, then to the opening
        of the summary, so an entry is never displayed as 'Untitled'."""
        if self.title:
            return self.title
        if self.source_ref:
            return self.source_ref
        head = (self.summary or "").strip().split("\n")[0]
        return (head[:80] + "...") if len(head) > 80 else (head or "Untitled")

    @property
    def tag_list(self) -> list[str]:
        return [t.strip() for t in (self.tags or "").split(",") if t.strip()]


class Link(db.Model):
    __tablename__ = "links"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    from_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), nullable=False, index=True)
    to_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), nullable=False, index=True)
    relationship_type = db.Column(db.String(20), nullable=False)
    rationale = db.Column(db.Text)
    similarity = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class RagChunk(db.Model):
    """Persisted, lazily-built RAG index -- replaces the old in-memory,
    TTL'd `_docs` cache in rag.py. Rebuilt on demand from
    HistoryEntry.source_text if ever missing (e.g. a row inserted before
    this table existed), never silently unavailable."""

    __tablename__ = "rag_chunks"

    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), nullable=False, index=True)
    chunk_index = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.LargeBinary, nullable=False)


class Watch(db.Model):
    __tablename__ = "watches"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    url = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(255))
    interval_minutes = db.Column(db.Integer, nullable=False, default=60)
    active = db.Column(db.Boolean, nullable=False, default=True)
    last_checked_at = db.Column(db.DateTime(timezone=True))
    last_content_hash = db.Column(db.String(64))
    last_snapshot_text = db.Column(db.Text)
    last_error = db.Column(db.Text)
    check_count = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class DigestItem(db.Model):
    __tablename__ = "digest_items"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    watch_id = db.Column(db.Integer, db.ForeignKey("watches.id"), nullable=False, index=True)
    headline = db.Column(db.Text, nullable=False)
    change_summary = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)

    watch = db.relationship("Watch")


class Flashcard(db.Model):
    __tablename__ = "flashcards"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    source_ref = db.Column(db.Text)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    ease_factor = db.Column(db.Float, nullable=False, default=2.5)
    interval_days = db.Column(db.Float, nullable=False, default=0)
    repetitions = db.Column(db.Integer, nullable=False, default=0)
    next_review_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)
    last_reviewed_at = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class Annotation(db.Model):
    __tablename__ = "annotations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), nullable=False, index=True)
    quote = db.Column(db.Text)
    note = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class Share(db.Model):
    __tablename__ = "shares"

    token = db.Column(db.String(64), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class Draft(db.Model):
    __tablename__ = "drafts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"))
    draft_type = db.Column(db.String(30), nullable=False)
    instructions = db.Column(db.Text)
    content = db.Column(db.Text, nullable=False)
    version = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class Job(db.Model):
    """Background-job status/results, now persisted (fixes the second
    half of the reported 'sessions don't get stored' bug -- this used to
    be an in-memory dict with a 30-minute TTL, wiped on every restart)."""

    __tablename__ = "jobs"

    id = db.Column(db.String(32), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    kind = db.Column(db.String(30))
    status = db.Column(db.String(20), nullable=False, default="running")
    progress = db.Column(db.Text)  # JSON-encoded list[str]
    result = db.Column(db.Text)  # JSON-encoded dict
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class Recording(db.Model):
    """A raw mic/webcam/video capture made in-browser, before (or
    instead of) being summarized -- kept so a recording is never lost
    just because its summarization step failed or hasn't run yet."""

    __tablename__ = "recordings"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    kind = db.Column(db.String(10), nullable=False)  # audio | image | video
    filename = db.Column(db.String(255), nullable=False)
    storage_path = db.Column(db.Text, nullable=False)
    mime_type = db.Column(db.String(100))
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class SavedSearch(db.Model):
    __tablename__ = "saved_searches"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    query = db.Column(db.Text, nullable=False)
    filters = db.Column(db.Text)  # JSON-encoded dict
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class Matter(db.Model):
    """A situation being documented.

    A tenancy, a contract, a workplace grievance, an insurance claim.
    Records belong to a matter rather than sitting in one undifferentiated
    pile, because the unit that gets disputed, disclosed and sealed is the
    matter, not the individual photograph.

    `reference` is a short human handle (M-0001) for use in
    correspondence -- "the records under M-0004" is something you can
    write in a letter; a database id is not.
    """

    __tablename__ = "matters"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    reference = db.Column(db.String(24), nullable=False, index=True)
    title = db.Column(db.String(300), nullable=False)
    kind = db.Column(db.String(40), nullable=False, default="other")
    counterparty = db.Column(db.String(300), default="")
    description = db.Column(db.Text, default="")
    status = db.Column(db.String(20), nullable=False, default="open")  # open | sealed | closed
    opened_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)
    closed_at = db.Column(db.DateTime(timezone=True))
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    __table_args__ = (db.UniqueConstraint("user_id", "reference", name="uq_matter_reference"),)


class MatterRecord(db.Model):
    """One sealed item of evidence, and its position in the chain.

    The ledger fields (`sequence`, `content_hash`, `prev_hash`,
    `entry_hash`) are written once at entry and never updated. Anything
    the user can revise afterwards -- and there is very little -- lives
    outside the hashed metadata, because a field that changes silently
    would break verification for an honest user and look like tampering.
    """

    __tablename__ = "matter_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    matter_id = db.Column(db.Integer, db.ForeignKey("matters.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), index=True)
    asset_id = db.Column(db.Integer, db.ForeignKey("source_assets.id"), index=True)

    record_uid = db.Column(db.String(40), nullable=False, index=True)
    sequence = db.Column(db.Integer, nullable=False)
    kind = db.Column(db.String(20), nullable=False, default="note")  # audio|video|image|pdf|text|url|note
    note = db.Column(db.Text, default="")
    occurred_at = db.Column(db.DateTime(timezone=True))  # when the event happened, if different from entry

    content_hash = db.Column(db.String(64), nullable=False)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    prev_hash = db.Column(db.String(64), nullable=False)
    entry_hash = db.Column(db.String(64), nullable=False, index=True)

    captured_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)

    matter = db.relationship("Matter", backref=db.backref("records", lazy="dynamic"))


class MatterSeal(db.Model):
    """A point-in-time commitment to everything in a matter.

    Sealing does not lock the matter -- records can still be added
    afterwards, producing a later seal. What a seal fixes is that the
    records present at that moment existed in that form at that moment,
    which is the claim that matters when someone alleges the account was
    assembled after the fact.
    """

    __tablename__ = "matter_seals"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    matter_id = db.Column(db.Integer, db.ForeignKey("matters.id"), nullable=False, index=True)
    record_count = db.Column(db.Integer, nullable=False, default=0)
    head_hash = db.Column(db.String(64), nullable=False)
    merkle_root = db.Column(db.String(64), nullable=False)
    manifest_json = db.Column(db.Text, nullable=False, default="{}")
    anchor_text = db.Column(db.Text, default="")
    anchored_note = db.Column(db.Text, default="")  # where the user says they published it
    sealed_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)


class SourceAsset(db.Model):
    """The original input, kept.

    Before this existed, summarizing a PDF or a recording threw the
    source away and kept only the text we extracted from it -- so once
    you navigated off the page, the actual file was gone and there was
    no way to listen to the recording again, re-read the PDF, or re-run
    the summary at a different detail level. That is the single most
    reported frustration with the app.

    Now every input is written to disk under instance/sources/<user_id>/
    and pointed at from here, so an entry can always be reopened with
    its source beside it. `sha256` lets us notice the same file being
    uploaded twice without comparing bytes every time."""

    __tablename__ = "source_assets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), index=True)
    kind = db.Column(db.String(16), nullable=False)  # text|pdf|audio|image|video|url
    filename = db.Column(db.String(255))
    storage_path = db.Column(db.Text)  # relative to instance/sources
    mime_type = db.Column(db.String(120))
    byte_size = db.Column(db.Integer, default=0)
    sha256 = db.Column(db.String(64), index=True)
    original_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class VaultItem(db.Model):
    """Ciphertext the server cannot read.

    The encryption key never reaches this process: it is derived in the
    browser from a passphrase the user alone knows (PBKDF2-SHA256,
    600k iterations) and the payload is sealed with AES-256-GCM before
    it is sent. What lands here is an opaque blob, its IV, its salt and
    a byte count. That is deliberate and it is the point -- a breach of
    this table, or a subpoena served on it, yields nothing readable.

    The tradeoff is equally real and we state it plainly in the UI: a
    forgotten passphrase means the data is gone. There is no reset,
    because a reset would prove we had access all along."""

    __tablename__ = "vault_items"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    item_key = db.Column(db.String(64), nullable=False, index=True)  # client-generated id
    ciphertext = db.Column(db.Text, nullable=False)
    iv = db.Column(db.String(64), nullable=False)
    salt = db.Column(db.String(64), nullable=False)
    algo = db.Column(db.String(40), nullable=False, default="AES-GCM-256")
    kdf_iterations = db.Column(db.Integer, nullable=False, default=600000)
    byte_size = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    __table_args__ = (db.UniqueConstraint("user_id", "item_key", name="uq_vault_user_item"),)


class CanvasLayout(db.Model):
    """Where the user put things on the spatial canvas. Stored as one
    JSON document per user per board: layouts are read and written whole,
    never queried field-by-field, so a row per node would buy nothing but
    joins."""

    __tablename__ = "canvas_layouts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False, default="Default board")
    payload = db.Column(db.Text, nullable=False, default="{}")  # {nodes, edges, view, groups}
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class ReaderSession(db.Model):
    """One speed-reading run. Kept because the adaptive pacing is only
    honest if it is driven by measured comprehension over time rather
    than by how fast the user *feels* they read."""

    __tablename__ = "reader_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), index=True)
    wpm = db.Column(db.Integer, nullable=False, default=300)
    words_read = db.Column(db.Integer, nullable=False, default=0)
    duration_ms = db.Column(db.Integer, nullable=False, default=0)
    comprehension = db.Column(db.Float)  # null when no quiz was taken
    completed = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)


class AutomationRule(db.Model):
    """A user-authored rule: when <trigger> and <condition>, do <actions>.

    `condition` is source text in the small expression language in
    app/core/rules.py. It is parsed and evaluated by our own interpreter,
    never by eval() -- these strings are user input that the server
    executes on a schedule, so treating them as code would be a remote
    execution hole with extra steps."""

    __tablename__ = "automation_rules"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    trigger = db.Column(db.String(60), nullable=False, default="entry.created")
    condition = db.Column(db.Text, nullable=False, default="")
    actions = db.Column(db.Text, nullable=False, default="[]")  # JSON list
    enabled = db.Column(db.Boolean, nullable=False, default=True)
    run_count = db.Column(db.Integer, nullable=False, default=0)
    match_count = db.Column(db.Integer, nullable=False, default=0)
    last_run_at = db.Column(db.DateTime(timezone=True))
    last_error = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class RuleRun(db.Model):
    """Audit trail for automation. If something ran while you weren't
    looking, you get to see exactly what it decided and why."""

    __tablename__ = "rule_runs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    rule_id = db.Column(db.Integer, db.ForeignKey("automation_rules.id"), index=True)
    matched = db.Column(db.Boolean, nullable=False, default=False)
    explain = db.Column(db.Text)
    actions_taken = db.Column(db.Text)  # JSON
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)


class SyncOp(db.Model):
    """One CRDT operation in the offline-sync log.

    Clients queue these locally while offline and push them on
    reconnect; the server keeps them so any other device can ask "what
    have I not seen?" and converge. Ops are immutable and idempotent --
    `op_id` is the client-generated identity that makes replaying the
    same operation harmless."""

    __tablename__ = "sync_ops"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    op_id = db.Column(db.String(64), nullable=False, index=True)
    replica_id = db.Column(db.String(64), nullable=False)
    entity = db.Column(db.String(40), nullable=False)
    entity_id = db.Column(db.String(64), nullable=False, index=True)
    field = db.Column(db.String(60), nullable=False, default="")
    action = db.Column(db.String(20), nullable=False)
    value = db.Column(db.Text)  # JSON-encoded
    lamport = db.Column(db.Integer, nullable=False, default=0)
    wall_clock = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)

    __table_args__ = (db.UniqueConstraint("user_id", "op_id", name="uq_syncop_user_op"),)


class AudioFingerprint(db.Model):
    """Acoustic landmarks for one recording, so the library can answer
    "have I heard this before?" without re-analysing every file. Stored
    as a compact JSON hash list -- see app/core/acoustics.py."""

    __tablename__ = "audio_fingerprints"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("history_entries.id"), index=True)
    asset_id = db.Column(db.Integer, db.ForeignKey("source_assets.id"), index=True)
    label = db.Column(db.String(255))
    fingerprint_id = db.Column(db.String(64), index=True)
    duration = db.Column(db.Float, default=0.0)
    hash_count = db.Column(db.Integer, default=0)
    hashes = db.Column(db.Text, nullable=False, default="[]")  # JSON [[hash, time], ...]
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)


class UserSession(db.Model):
    """One logged-in device.

    Flask-Login alone gives you a signed cookie and no way to answer
    "where am I currently logged in?" or "log me out of the laptop I
    left at the office". This table is that answer: a row per active
    session, refreshed on use, revocable individually or all at once."""

    __tablename__ = "user_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    session_key = db.Column(db.String(64), nullable=False, unique=True, index=True)
    user_agent = db.Column(db.String(400))
    ip_address = db.Column(db.String(64))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    last_seen_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    revoked = db.Column(db.Boolean, nullable=False, default=False)


class CacheEntry(db.Model):
    """Response cache for summarize_text / embed_text / the weekly
    digest narrative. Not user-scoped on purpose: the cache key already
    hashes the exact input, and the same input produces the same Gemini
    output regardless of who asked -- sharing hits across accounts is a
    feature (fewer calls, lower latency), not a leak, since only the
    result is cached, never which user requested it."""

    __tablename__ = "cache_entries"

    cache_key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.Float, nullable=False)
