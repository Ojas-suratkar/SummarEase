"""
Tamper-evident record keeping.

The problem
-----------
Evidence loses arguments not because it is weak but because it is
deniable. A photograph has a file date that any phone can change. A
recording can be trimmed. A note written afterwards is indistinguishable
from one written at the time. So the other side says "that was edited",
"that was taken later", "that is out of context", and a true account
loses to a confident denial.

This module makes alteration detectable. Every record is hashed on
entry, and every entry commits to the one before it, so the records form
a chain. Changing the content of record 4 changes its hash, which
breaks record 5's commitment, which breaks 6, and so on to the end. The
only way to alter history without detection is to rewrite every
subsequent entry, and if the head of the chain has been published or
handed to anyone, even that fails.

What this does and does not prove
---------------------------------
It is worth being exact, because overstating it would be the fastest way
to get someone hurt in a real dispute.

It DOES prove:
  * That the set of records has not been altered, reordered, or had
    entries removed or inserted since they were sealed.
  * That a given record belongs to a sealed set, via a Merkle proof,
    without having to disclose the other records in that set.
  * The order in which records were entered, relative to each other.

It does NOT prove:
  * That the content is true. A sealed lie is still a lie.
  * That a record was created when its timestamp says, unless the chain
    head was anchored somewhere outside this system (see `anchor_text`).
    Absent an external anchor, the timestamps are this system's own
    assertion, and an adversary with control of the server could have
    produced the whole chain yesterday.
  * Anything about who created the record beyond which account entered it.

The honest summary is that this converts "you could have made this up
afterwards" from an unanswerable accusation into a specific, checkable
claim. That is a smaller thing than proof, and a much larger thing than
what a folder of loose photographs offers.

Independence
------------
The verification here is intentionally simple, uses nothing but
`hashlib` and `json` from the standard library, and is duplicated in the
standalone verifier shipped inside every exported pack. A third party
must never have to run our software, or trust it, to check our claims.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Bumped only if the hashing scheme changes in a way that would make old
# chains verify differently. Written into every sealed manifest so a
# verifier reading a five-year-old pack knows which rules to apply.
LEDGER_VERSION = "1"

GENESIS_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Canonical encoding
#
# Two systems must derive byte-identical input for the same record, or
# their hashes disagree and honest records look tampered with. JSON is
# permissive about key order, whitespace and unicode escaping, so the
# encoding is pinned here rather than left to whatever the default is on
# the machine doing the check.
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_bytes(data: bytes) -> str:
    """Content hash of a file. Used for the originals: photographs,
    recordings, documents."""
    return sha256_hex(data)


def hash_text(text: str) -> str:
    return sha256_hex((text or "").encode("utf-8"))


def hash_stream(chunks: Iterable[bytes]) -> str:
    """Hash without holding the whole file in memory. A two-hour video
    should not require two hours of video in RAM to seal."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    """One sealed record.

    `content_hash` covers the thing itself -- the file bytes, or the text
    if there is no file. `metadata` covers the circumstances of entry:
    when, by which account, from which device, what kind of record, and
    the note the person wrote at the time. Both are inside `entry_hash`,
    so changing the note after the fact breaks the chain exactly as
    changing the photograph would. That is deliberate: in a dispute, when
    someone wrote a description matters as much as what it says.
    """

    sequence: int
    record_id: str
    content_hash: str
    metadata: dict = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def compute_hash(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "v": LEDGER_VERSION,
                    "sequence": self.sequence,
                    "record_id": str(self.record_id),
                    "content_hash": self.content_hash,
                    "metadata": self.metadata,
                    "prev_hash": self.prev_hash,
                }
            )
        )

    def seal(self) -> "LedgerEntry":
        self.entry_hash = self.compute_hash()
        return self

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "record_id": str(self.record_id),
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    @staticmethod
    def from_dict(data: dict) -> "LedgerEntry":
        return LedgerEntry(
            sequence=int(data.get("sequence", 0)),
            record_id=str(data.get("record_id", "")),
            content_hash=str(data.get("content_hash", "")),
            metadata=data.get("metadata") or {},
            prev_hash=str(data.get("prev_hash", GENESIS_HASH)),
            entry_hash=str(data.get("entry_hash", "")),
        )


def append_entry(
    previous: LedgerEntry | None,
    *,
    record_id: str,
    content_hash: str,
    metadata: dict | None = None,
) -> LedgerEntry:
    """Add one record to the end of a chain."""
    entry = LedgerEntry(
        sequence=(previous.sequence + 1) if previous else 1,
        record_id=str(record_id),
        content_hash=content_hash,
        metadata=metadata or {},
        prev_hash=previous.entry_hash if previous else GENESIS_HASH,
    )
    return entry.seal()


def build_chain(records: Sequence[dict]) -> list[LedgerEntry]:
    """Build a chain from scratch. `records` are dicts with `record_id`,
    `content_hash` and `metadata`, in the order they were entered."""
    chain: list[LedgerEntry] = []
    previous: LedgerEntry | None = None
    for record in records:
        previous = append_entry(
            previous,
            record_id=record["record_id"],
            content_hash=record["content_hash"],
            metadata=record.get("metadata") or {},
        )
        chain.append(previous)
    return chain


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_chain(entries: Sequence[LedgerEntry | dict], *, content_hashes: dict[str, str] | None = None) -> dict:
    """Recompute the whole chain and report precisely what is wrong.

    A verifier that answers only "valid" or "invalid" is close to useless
    in an argument: the other side will say the tool is broken. This
    names the record, the sequence number, and the nature of the
    discrepancy, so the finding can be examined rather than believed.

    `content_hashes` maps record_id to the hash of the file as it exists
    *now*. Supplying it additionally checks that the stored originals
    still match what was sealed -- catching a file replaced on disk,
    which the chain alone would not notice.
    """
    normalised = [e if isinstance(e, LedgerEntry) else LedgerEntry.from_dict(e) for e in entries]
    problems: list[dict] = []

    if not normalised:
        return {
            "valid": True,
            "entries": 0,
            "problems": [],
            "head": None,
            "summary": "The record is empty. There is nothing to verify.",
        }

    expected_prev = GENESIS_HASH
    expected_sequence = 1

    for entry in normalised:
        if entry.sequence != expected_sequence:
            problems.append(
                {
                    "record_id": entry.record_id,
                    "sequence": entry.sequence,
                    "kind": "sequence_gap",
                    "detail": f"Expected record number {expected_sequence} but found {entry.sequence}. "
                    "A record has been removed or inserted.",
                }
            )

        if entry.prev_hash != expected_prev:
            problems.append(
                {
                    "record_id": entry.record_id,
                    "sequence": entry.sequence,
                    "kind": "broken_link",
                    "detail": "This record does not point at the one before it. Either an earlier "
                    "record was altered, or the order has been changed.",
                }
            )

        recomputed = entry.compute_hash()
        if entry.entry_hash != recomputed:
            problems.append(
                {
                    "record_id": entry.record_id,
                    "sequence": entry.sequence,
                    "kind": "altered_entry",
                    "detail": "The sealed fingerprint does not match this record's current contents. "
                    "The record itself, its description, or its timestamp has been changed since it was sealed.",
                    "expected": recomputed,
                    "found": entry.entry_hash,
                }
            )

        if content_hashes is not None:
            current = content_hashes.get(str(entry.record_id))
            if current is None:
                problems.append(
                    {
                        "record_id": entry.record_id,
                        "sequence": entry.sequence,
                        "kind": "missing_file",
                        "detail": "The original file for this record is no longer present.",
                    }
                )
            elif current != entry.content_hash:
                problems.append(
                    {
                        "record_id": entry.record_id,
                        "sequence": entry.sequence,
                        "kind": "altered_file",
                        "detail": "The original file has been replaced or modified since it was sealed.",
                        "expected": entry.content_hash,
                        "found": current,
                    }
                )

        expected_prev = entry.entry_hash
        expected_sequence = entry.sequence + 1

    valid = not problems
    if valid:
        summary = (
            f"All {len(normalised)} records verified. The chain is unbroken, nothing has been "
            "added, removed, reordered or altered since sealing."
        )
    else:
        affected = sorted({p["sequence"] for p in problems})
        summary = (
            f"{len(problems)} problem(s) found across record(s) {', '.join(str(a) for a in affected)}. "
            "This record set has been altered since it was sealed and should not be relied on."
        )

    return {
        "valid": valid,
        "entries": len(normalised),
        "problems": problems,
        "head": normalised[-1].entry_hash,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Merkle tree
#
# The chain proves the set is intact, but proving one record is in it
# means handing over every record so the chain can be recomputed. In a
# real dispute that is often unacceptable: disclosing eleven unrelated
# private recordings to prove the twelfth is genuine is a bad trade.
#
# A Merkle tree fixes that. Each record sits at a leaf, pairs are hashed
# together up to a single root, and membership can be shown with only
# the handful of sibling hashes along one path. The others stay private
# and are revealed as nothing but hashes.
# ---------------------------------------------------------------------------


def _hash_pair(left: str, right: str) -> str:
    # The 0x01 prefix distinguishes internal nodes from leaves (prefixed
    # 0x00 below). Without that separation an attacker can present an
    # internal node as if it were a leaf -- the classic second-preimage
    # weakness in naive Merkle implementations.
    return sha256_hex(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


def _hash_leaf(value: str) -> str:
    return sha256_hex(b"\x00" + value.encode("utf-8"))


def merkle_root(leaves: Sequence[str]) -> str:
    """Root over the entry hashes. An odd node at any level is promoted
    unchanged rather than duplicated; duplicating it would let two
    different record sets produce the same root."""
    if not leaves:
        return GENESIS_HASH
    level = [_hash_leaf(leaf) for leaf in leaves]
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_hash_pair(level[i], level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def merkle_proof(leaves: Sequence[str], index: int) -> list[dict]:
    """The sibling hashes needed to walk one leaf up to the root."""
    if not leaves or index < 0 or index >= len(leaves):
        return []

    proof: list[dict] = []
    level = [_hash_leaf(leaf) for leaf in leaves]
    position = index

    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level) - 1, 2):
            left, right = level[i], level[i + 1]
            if i == position - (position % 2) and position < len(level) - (len(level) % 2):
                if position % 2 == 0:
                    proof.append({"side": "right", "hash": right})
                else:
                    proof.append({"side": "left", "hash": left})
            nxt.append(_hash_pair(left, right))
        if len(level) % 2 == 1:
            if position == len(level) - 1:
                pass  # promoted unchanged; no sibling to record
            nxt.append(level[-1])
        position //= 2
        level = nxt

    return proof


def verify_merkle_proof(leaf: str, proof: Sequence[dict], root: str) -> bool:
    """Check one record against a published root, given only its
    siblings. This is what lets someone confirm a single recording
    belongs to a sealed set without seeing the rest of it."""
    current = _hash_leaf(leaf)
    for step in proof:
        sibling = step.get("hash", "")
        if step.get("side") == "left":
            current = _hash_pair(sibling, current)
        else:
            current = _hash_pair(current, sibling)
    return current == root


# ---------------------------------------------------------------------------
# Sealing and anchoring
# ---------------------------------------------------------------------------


def seal_manifest(entries: Sequence[LedgerEntry], *, matter: dict, sealed_at: str) -> dict:
    """The document that travels with an exported pack."""
    leaves = [e.entry_hash for e in entries]
    return {
        "ledger_version": LEDGER_VERSION,
        "algorithm": "SHA-256",
        "matter": matter,
        "sealed_at": sealed_at,
        "record_count": len(entries),
        "head_hash": entries[-1].entry_hash if entries else GENESIS_HASH,
        "merkle_root": merkle_root(leaves),
        "entries": [e.to_dict() for e in entries],
    }


def anchor_text(manifest: dict) -> str:
    """A short, human-copyable commitment to the whole record set.

    The single most valuable thing someone can do with a record they may
    later need is to put this string somewhere outside their own control
    and outside this system, at the time -- email it to their solicitor,
    post it publicly, send it to themselves. It reveals nothing about
    the contents, and it fixes the record as of that moment: anything
    produced later cannot match a commitment that already exists
    elsewhere.

    This is the step that turns "the timestamps are your own software's
    word" into something an opponent cannot argue with, and it costs
    nothing but the discipline of doing it.
    """
    return (
        f"SummarEase record seal\n"
        f"Matter: {manifest['matter'].get('reference') or manifest['matter'].get('title', '')}\n"
        f"Records: {manifest['record_count']}\n"
        f"Sealed: {manifest['sealed_at']}\n"
        f"Root: {manifest['merkle_root']}\n"
        f"Head: {manifest['head_hash']}\n"
        f"Algorithm: SHA-256, ledger v{manifest['ledger_version']}"
    )


def fingerprint_short(hash_hex: str) -> str:
    """A shortened, readable form for screen and print. Full hashes are
    unreadable to a person and get transcribed wrongly; grouped hex is
    checkable by eye. The full value is always kept alongside for
    machine comparison -- this is for humans, never for verification."""
    clean = (hash_hex or "").strip().lower()
    if len(clean) < 16:
        return clean
    return f"{clean[:4]}-{clean[4:8]}-{clean[8:12]}-{clean[12:16]}".upper()


if __name__ == "__main__":  # pragma: no cover
    import sys

    passed = failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"PASS  {label}" + (f"  -- {detail}" if detail else ""))
        else:
            failed += 1
            print(f"FAIL  {label}" + (f"  -- {detail}" if detail else ""))

    records = [
        {"record_id": "r1", "content_hash": hash_text("photo of the kitchen wall"), "metadata": {"kind": "image", "note": "crack above the socket", "at": "2026-01-04T09:00:00Z"}},
        {"record_id": "r2", "content_hash": hash_text("recording of the call"), "metadata": {"kind": "audio", "note": "agent agreed it was pre-existing", "at": "2026-01-04T14:30:00Z"}},
        {"record_id": "r3", "content_hash": hash_text("tenancy agreement pdf"), "metadata": {"kind": "pdf", "note": "signed copy", "at": "2026-01-05T10:00:00Z"}},
        {"record_id": "r4", "content_hash": hash_text("email thread"), "metadata": {"kind": "text", "note": "refusal to return deposit", "at": "2026-03-02T16:05:00Z"}},
    ]

    chain = build_chain(records)
    check("chain builds", len(chain) == 4)
    check("first entry points at genesis", chain[0].prev_hash == GENESIS_HASH)
    check("each entry links to the last", all(chain[i].prev_hash == chain[i - 1].entry_hash for i in range(1, 4)))

    result = verify_chain(chain)
    check("clean chain verifies", result["valid"], result["summary"])

    # Tamper with content.
    tampered = [LedgerEntry.from_dict(e.to_dict()) for e in chain]
    tampered[1].content_hash = hash_text("a different recording")
    result = verify_chain(tampered)
    check("altered content detected", not result["valid"])
    check("names the altered record", any(p["kind"] == "altered_entry" and p["sequence"] == 2 for p in result["problems"]),
          str([p["kind"] for p in result["problems"]]))

    # Tamper with the note only -- the file is untouched.
    note_tampered = [LedgerEntry.from_dict(e.to_dict()) for e in chain]
    note_tampered[0].metadata = dict(note_tampered[0].metadata, note="damage was NOT pre-existing")
    result = verify_chain(note_tampered)
    check("altered description detected", not result["valid"])

    # Remove a record from the middle.
    removed = [LedgerEntry.from_dict(e.to_dict()) for e in chain]
    del removed[2]
    result = verify_chain(removed)
    check("removed record detected", not result["valid"])
    check("reports a gap or broken link",
          any(p["kind"] in ("sequence_gap", "broken_link") for p in result["problems"]))

    # Reorder.
    reordered = [LedgerEntry.from_dict(e.to_dict()) for e in chain]
    reordered[1], reordered[2] = reordered[2], reordered[1]
    check("reordering detected", not verify_chain(reordered)["valid"])

    # Replace the file on disk without touching the ledger.
    live = {r["record_id"]: r["content_hash"] for r in records}
    check("matching files verify", verify_chain(chain, content_hashes=live)["valid"])
    live["r3"] = hash_text("a substituted document")
    result = verify_chain(chain, content_hashes=live)
    check("swapped file detected", not result["valid"])
    check("reports the swapped file", any(p["kind"] == "altered_file" for p in result["problems"]))

    # A full rewrite -- the honest limit of a chain with no external anchor.
    rewritten = build_chain(
        [{"record_id": r["record_id"],
          "content_hash": hash_text("fabricated") if r["record_id"] == "r2" else r["content_hash"],
          "metadata": r["metadata"]} for r in records]
    )
    check("a complete rewrite still self-verifies (why anchoring matters)", verify_chain(rewritten)["valid"])
    check("but the rewrite has a different head", rewritten[-1].entry_hash != chain[-1].entry_hash)
    check("and a different root", merkle_root([e.entry_hash for e in rewritten]) != merkle_root([e.entry_hash for e in chain]))

    # Merkle membership.
    leaves = [e.entry_hash for e in chain]
    root = merkle_root(leaves)
    for i in range(len(leaves)):
        proof = merkle_proof(leaves, i)
        if not verify_merkle_proof(leaves[i], proof, root):
            check(f"merkle proof for leaf {i}", False)
            break
    else:
        check("every record proves membership without disclosing the others", True, f"{len(leaves)} leaves")

    check("a foreign record fails the membership check",
          not verify_merkle_proof(hash_text("never entered"), merkle_proof(leaves, 0), root))

    # Odd leaf counts are where naive Merkle code breaks.
    for size in range(1, 18):
        synthetic = [hash_text(f"x{i}") for i in range(size)]
        r = merkle_root(synthetic)
        ok = all(verify_merkle_proof(synthetic[i], merkle_proof(synthetic, i), r) for i in range(size))
        if not ok:
            check(f"merkle handles {size} leaves", False)
            break
    else:
        check("merkle correct for every set size from 1 to 17", True)

    # Determinism across key order.
    a = LedgerEntry(1, "r", "c", {"b": 2, "a": 1}, GENESIS_HASH).seal()
    b = LedgerEntry(1, "r", "c", {"a": 1, "b": 2}, GENESIS_HASH).seal()
    check("hashing is independent of key order", a.entry_hash == b.entry_hash)

    check("unicode survives canonical encoding",
          LedgerEntry(1, "r", hash_text("café — naïve 日本"), {}, GENESIS_HASH).seal().entry_hash ==
          LedgerEntry(1, "r", hash_text("café — naïve 日本"), {}, GENESIS_HASH).seal().entry_hash)

    check("empty ledger verifies cleanly", verify_chain([])["valid"])
    check("readable fingerprint is grouped", fingerprint_short("a" * 64) == "AAAA-AAAA-AAAA-AAAA")

    manifest = seal_manifest(chain, matter={"title": "Flat 4B deposit", "reference": "M-0001"}, sealed_at="2026-03-03T10:00:00Z")
    check("manifest carries a root and head", bool(manifest["merkle_root"]) and bool(manifest["head_hash"]))
    check("anchor text contains the root", manifest["merkle_root"] in anchor_text(manifest))

    print()
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
