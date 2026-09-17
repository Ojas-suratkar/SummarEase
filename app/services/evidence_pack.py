"""
Exporting a matter as a self-contained, independently checkable pack.

This is the part of the product that has to be beyond reproach, because
it is the artefact that leaves our hands and gets argued about. Someone
on the other side of a dispute -- a solicitor, a claims handler, an
adjudicator, an opposing party who assumes you are lying -- has to be
able to open it and satisfy themselves, without installing anything,
without an internet connection, and above all without taking our word
for any of it.

So the pack ships its own verifier: a single Python file using nothing
but the standard library, short enough to read in five minutes and
confirm it does what it claims. If our software and that script ever
disagreed, the script would be right, because it is the one the other
side can audit.

Contents
--------
    README.txt          what this is, and how to check it
    verify.py           standalone verifier, standard library only
    manifest.json       the sealed chain
    index.html          readable report, opens in any browser
    records/            the original files, named by sequence

Deliberately absent: anything that requires our server, our account
system, or an internet connection. A pack handed over on a USB stick in
2031 has to still work.
"""
from __future__ import annotations

import html
import io
import json
import logging
import zipfile
from datetime import datetime, timezone

from ..core import ledger
from ..models import MatterRecord, SourceAsset
from . import matters as matters_service
from . import source_store

logger = logging.getLogger(__name__)


# The verifier that travels with every pack. Kept as a literal so it
# cannot drift from what we ship, and written to be read by a sceptic:
# no imports beyond hashlib/json/os/sys, no cleverness, no dependencies.
_VERIFIER_SOURCE = '''#!/usr/bin/env python3
"""
Independent verifier for a SummarEase record pack.

Run it:      python3 verify.py
Requires:    Python 3.8 or later. Nothing else. No internet connection.

What it checks
--------------
1. That every record's sealed fingerprint matches its contents.
2. That each record commits to the one before it, so nothing has been
   inserted, removed or reordered.
3. That each original file on disk still hashes to what was sealed.
4. That the Merkle root matches the one in the manifest.

What a pass means
-----------------
The records are exactly as they were when sealed. It does NOT mean the
contents are true, and it does not by itself prove when they were made
-- for that, compare the root below against a copy of the seal that was
published somewhere independent at the time.

This script is deliberately short so that you can read it and satisfy
yourself it does what it says.
"""
import hashlib
import json
import os
import sys

GENESIS = "0" * 64


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def entry_hash(entry):
    return sha256_hex(canonical({
        "v": entry.get("_ledger_version", "1"),
        "sequence": entry["sequence"],
        "record_id": str(entry["record_id"]),
        "content_hash": entry["content_hash"],
        "metadata": entry["metadata"],
        "prev_hash": entry["prev_hash"],
    }))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_leaf(value):
    return sha256_hex(b"\\x00" + value.encode("utf-8"))


def hash_pair(left, right):
    return sha256_hex(b"\\x01" + bytes.fromhex(left) + bytes.fromhex(right))


def merkle_root(leaves):
    if not leaves:
        return GENESIS
    level = [hash_leaf(v) for v in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hash_pair(level[i], level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(here, "manifest.json")
    if not os.path.exists(manifest_path):
        print("FAIL: manifest.json is missing. This pack is incomplete.")
        return 2

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    version = manifest.get("ledger_version", "1")
    entries = manifest.get("entries", [])
    files = manifest.get("files", {})

    print("Record pack verification")
    print("=" * 62)
    print("Matter:   %s (%s)" % (manifest.get("matter", {}).get("title", "-"),
                                 manifest.get("matter", {}).get("reference", "-")))
    print("Sealed:   %s" % manifest.get("sealed_at", "-"))
    print("Records:  %d" % len(entries))
    print("Hash:     %s" % manifest.get("algorithm", "SHA-256"))
    print("=" * 62)
    print("")

    problems = []
    expected_prev = GENESIS
    expected_seq = 1

    for entry in entries:
        entry["_ledger_version"] = version
        seq = entry.get("sequence")
        label = "Record %s" % seq

        if seq != expected_seq:
            problems.append("%s: expected number %d, found %s -- a record was "
                            "removed or inserted." % (label, expected_seq, seq))

        if entry.get("prev_hash") != expected_prev:
            problems.append("%s: does not link to the previous record -- the order "
                            "changed or an earlier record was altered." % label)

        recomputed = entry_hash(entry)
        if recomputed != entry.get("entry_hash"):
            problems.append("%s: contents do not match the sealed fingerprint -- "
                            "the record, its description or its timestamp was changed." % label)

        stored = files.get(str(entry.get("record_id")))
        if stored:
            path = os.path.join(here, stored)
            if not os.path.exists(path):
                problems.append("%s: the original file %s is missing." % (label, stored))
            else:
                actual = file_hash(path)
                if actual != entry.get("content_hash"):
                    problems.append("%s: the file %s does not match what was sealed -- "
                                    "it has been replaced or modified." % (label, stored))

        expected_prev = entry.get("entry_hash")
        expected_seq = (seq or 0) + 1

    computed_root = merkle_root([e.get("entry_hash", "") for e in entries])
    stated_root = manifest.get("merkle_root", "")
    if computed_root != stated_root:
        problems.append("The overall root does not match the manifest. The set of "
                        "records is not the set that was sealed.")

    for entry in entries:
        seq = entry.get("sequence")
        note = entry.get("metadata", {}).get("note", "")
        kind = entry.get("metadata", {}).get("kind", "")
        when = entry.get("metadata", {}).get("occurred_at", "")
        print("  %3s  %-10s %-20s %s" % (seq, kind, when[:19], note[:44]))

    print("")
    print("=" * 62)
    if problems:
        print("RESULT: FAILED -- %d problem(s) found." % len(problems))
        print("")
        for problem in problems:
            print("  * %s" % problem)
        print("")
        print("This pack has been altered since it was sealed.")
        return 1

    print("RESULT: PASSED")
    print("")
    print("All %d records are intact. Nothing has been added, removed," % len(entries))
    print("reordered or altered since the pack was sealed.")
    print("")
    print("Root fingerprint:")
    print("  %s" % stated_root)
    print("")
    print("If a copy of this root was published independently at the time")
    print("of sealing, compare it against the value above. A match shows")
    print("the records existed in this exact form on that date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


_README = """SummarEase record pack
======================

Matter:    {title} ({reference})
Sealed:    {sealed_at}
Records:   {count}
Root:      {root}

WHAT THIS IS
------------
A set of records -- photographs, recordings, documents and written
statements -- that were entered one at a time and sealed so that any
later alteration becomes detectable.

Each record was hashed when it was entered, and each entry commits to
the one before it. Changing any record, its description or its
timestamp changes its fingerprint, which breaks every record after it.

HOW TO CHECK IT YOURSELF
------------------------
You do not need to trust the software that produced this pack, and you
do not need an internet connection.

    python3 verify.py

The script uses only the Python standard library and is short enough to
read in full before you run it. It recomputes every fingerprint from the
files in this folder and reports any discrepancy, naming the specific
record affected.

WHAT A PASS PROVES
------------------
That these records have not been altered, reordered, or had entries
added or removed since the seal date shown above.

WHAT IT DOES NOT PROVE
----------------------
That the contents are true. A sealed statement is still only a
statement.

That the records were made on the dates they claim -- unless the root
fingerprint above was published somewhere independent at the time of
sealing. If it was, compare the two values: a match establishes that
these records existed in this exact form on that date, because a
commitment already in someone else's hands cannot be produced after the
fact.

CONTENTS
--------
    README.txt      this file
    verify.py       the independent verifier
    manifest.json   the sealed chain, machine readable
    index.html      a readable report, opens in any browser
    records/        the original files, numbered in entry order
"""


def _safe_component(value: str) -> str:
    keep = "".join(c if c.isalnum() or c in "._- " else "_" for c in (value or ""))
    return keep.strip().replace(" ", "_")[:60] or "record"


def build_pack(user_id: int, matter_id: int) -> tuple[bytes, str] | None:
    """Assemble the ZIP. Returns (bytes, filename) or None."""
    matter = matters_service.get_matter(user_id, matter_id)
    if matter is None:
        return None

    rows = (
        MatterRecord.query.filter_by(matter_id=matter_id, user_id=user_id)
        .order_by(MatterRecord.sequence.asc())
        .all()
    )
    if not rows:
        return None

    entries = []
    file_map: dict[str, str] = {}
    payloads: list[tuple[str, bytes]] = []

    for row in rows:
        metadata = json.loads(row.metadata_json or "{}")
        entries.append(
            ledger.LedgerEntry(
                sequence=row.sequence,
                record_id=row.record_uid,
                content_hash=row.content_hash,
                metadata=metadata,
                prev_hash=row.prev_hash,
                entry_hash=row.entry_hash,
            )
        )

        if row.asset_id:
            asset = SourceAsset.query.filter_by(id=row.asset_id, user_id=user_id).first()
            path = source_store.absolute_path(asset) if asset else None
            if path is not None:
                name = f"records/{row.sequence:03d}-{_safe_component(asset.filename or row.kind)}"
                file_map[row.record_uid] = name
                try:
                    payloads.append((name, path.read_bytes()))
                except OSError as exc:
                    logger.warning("Could not read asset for record %s: %s", row.id, exc)

    sealed_at = datetime.now(timezone.utc).isoformat()
    manifest = ledger.seal_manifest(
        entries,
        matter={
            "title": matter.title,
            "reference": matter.reference,
            "kind": matter.kind,
            "counterparty": matter.counterparty or "",
        },
        sealed_at=sealed_at,
    )
    manifest["files"] = file_map

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.txt",
            _README.format(
                title=matter.title,
                reference=matter.reference,
                sealed_at=sealed_at,
                count=len(entries),
                root=manifest["merkle_root"],
            ),
        )
        archive.writestr("verify.py", _VERIFIER_SOURCE)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("index.html", _render_report(matter, entries, manifest, file_map))
        for name, data in payloads:
            archive.writestr(name, data)

    filename = f"{matter.reference}-{_safe_component(matter.title)}-pack.zip"
    return buffer.getvalue(), filename


def _render_report(matter, entries, manifest: dict, file_map: dict) -> str:
    """A readable report that opens in any browser with no server.

    Deliberately styled inline and using no external resources -- a
    report that needs a CDN is a report that stops working the moment
    someone opens it offline or in five years.
    """
    rows = []
    for entry in entries:
        metadata = entry.metadata or {}
        filename = file_map.get(entry.record_id, "")
        link = (
            f'<a href="{html.escape(filename)}">{html.escape(filename.split("/")[-1])}</a>'
            if filename else '<span class="muted">no file</span>'
        )
        rows.append(
            f"""<tr>
  <td class="num">{entry.sequence}</td>
  <td>{html.escape(metadata.get("kind", ""))}</td>
  <td>{html.escape((metadata.get("occurred_at") or "")[:19].replace("T", " "))}</td>
  <td>{html.escape(metadata.get("note", ""))}</td>
  <td>{link}</td>
  <td class="hash">{html.escape(entry.entry_hash[:16])}…</td>
</tr>"""
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(matter.reference)} — record pack</title>
<style>
  body {{ font-family: Georgia, "Times New Roman", serif; max-width: 60rem; margin: 2.5rem auto;
         padding: 0 1.5rem; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
  .ref {{ color: #666; font-size: 0.95rem; margin-bottom: 1.8rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.5rem 0; font-size: 0.9rem; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.55rem 0.5rem; text-align: left; vertical-align: top; }}
  th {{ border-bottom: 2px solid #333; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  .num {{ width: 3rem; color: #666; }}
  .hash, .rootbox {{ font-family: "SF Mono", Menlo, Consolas, monospace; }}
  .hash {{ font-size: 0.8rem; color: #666; }}
  .muted {{ color: #999; }}
  .rootbox {{ background: #f4f4f4; border: 1px solid #ddd; padding: 0.9rem; word-break: break-all;
              font-size: 0.85rem; margin: 1rem 0; }}
  .note {{ border-left: 3px solid #999; padding-left: 1rem; color: #444; margin: 1.8rem 0; }}
  @media print {{ body {{ margin: 0; }} }}
</style></head><body>

<h1>{html.escape(matter.title)}</h1>
<div class="ref">
  Reference {html.escape(matter.reference)} &middot; {html.escape(matter.counterparty or "no counterparty recorded")}
  &middot; sealed {html.escape(manifest["sealed_at"][:19].replace("T", " "))} UTC
  &middot; {len(entries)} records
</div>

<p>Each record below was hashed when entered and committed to the record before it.
Altering any of them — including a description or a timestamp — breaks the chain and
is detectable.</p>

<table>
  <thead><tr><th>#</th><th>Kind</th><th>When</th><th>Description</th><th>File</th><th>Fingerprint</th></tr></thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>

<h2 style="font-size:1.1rem;">Root fingerprint</h2>
<div class="rootbox">{html.escape(manifest["merkle_root"])}</div>

<div class="note">
  <p><strong>To verify this pack independently</strong>, run <code>python3 verify.py</code>
  from this folder. It uses only the Python standard library and needs no internet
  connection. It recomputes every fingerprint from the files present here and names any
  record that does not match.</p>

  <p><strong>What a pass proves:</strong> that these records have not been altered,
  reordered, or had entries added or removed since the seal date above.</p>

  <p><strong>What it does not prove:</strong> that the contents are true, or that the
  records were made when they claim — unless the root fingerprint above was published
  somewhere independent at the time of sealing, in which case a match establishes that
  these records existed in this exact form on that date.</p>
</div>

</body></html>"""
