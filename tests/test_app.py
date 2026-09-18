"""
End-to-end tests for the application.

Run from the repository root:

    python tests/test_app.py

Deliberately dependency-free -- no pytest, no fixtures library -- so it
runs anywhere the application itself runs. Everything happens against a
throwaway SQLite database in a temporary directory, so running the tests
never touches real data.

GEMINI_API_KEY is explicitly unset before the application is imported.
That is not an oversight: it proves the parts of this application that
are meant to work without an AI service actually do, which is a claim the
README makes and which would otherwise go unchecked.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Run from the repository root regardless of where this was invoked.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_TMP = tempfile.mkdtemp(prefix="summarease-tests-")
os.environ["FLASK_SECRET_KEY"] = "testing-only-not-a-real-secret"
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP, "test.db")
os.environ.pop("GEMINI_API_KEY", None)

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import HistoryEntry, Job, User  # noqa: E402
from app.services import source_store  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  pass  {label}" + (f"  [{detail}]" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")


app = create_app()
client = app.test_client()


# ---------------------------------------------------------------------------
section("Accounts")
# ---------------------------------------------------------------------------

response = client.post(
    "/signup",
    data={"email": "tester@example.com", "display_name": "Tester",
          "password": "a-long-enough-password", "confirm_password": "a-long-enough-password"},
    follow_redirects=True,
)
check("an account can be created", response.status_code == 200)

with app.app_context():
    user = User.query.filter_by(email="tester@example.com").first()
    check("the password is hashed, never stored as written",
          user.password_hash != "a-long-enough-password" and len(user.password_hash) > 40)
    user_id = user.id

    first = HistoryEntry(
        user_id=user_id, source_type="article", source_ref="https://example.com/report",
        summary="The platform reached 4.2 million users in Q3 2024.",
        source_text=("The platform reached 4.2 million users in Q3 2024. "
                     "Revenue was $12M last year. The company was founded in 1997."),
    )
    second = HistoryEntry(
        user_id=user_id, source_type="pdf", source_ref="analyst-note.pdf",
        summary="Analysts put the user base at 6.1 million and revenue at EUR 12M.",
        source_text="Analysts put the user base at 6.1 million users. Revenue reached EUR 12M.",
    )
    db.session.add_all([first, second])
    db.session.commit()
    first_id, second_id = first.id, second.id

    with app.test_request_context():
        asset = source_store.store_bytes(
            user_id, b"%PDF-1.4 stand-in for a real document", kind="pdf",
            filename="analyst-note.pdf", mime_type="application/pdf", entry_id=second_id,
        )
        asset_id = asset.id

    db.session.add(Job(id="test-job", user_id=user_id, kind="pdf", status="running",
                       progress=json.dumps(["Extracting text...", "Summarising..."])))
    db.session.commit()


# ---------------------------------------------------------------------------
section("Pages render")
# ---------------------------------------------------------------------------

for path in ["/", "/add", "/analyse", "/toolkit", "/library", "/timeline",
             "/contradictions", "/trace", "/settings/security", f"/entry/{first_id}"]:
    check(f"GET {path}", client.get(path).status_code == 200)


# ---------------------------------------------------------------------------
section("Analysis works with no AI service configured")
# ---------------------------------------------------------------------------

timeline = client.get("/api/timeline").get_json()
check("dates are found across saved sources", timeline["total"] >= 2, f"{timeline['total']} events")
check("a quarter resolves to the right year",
      any(e["start_date"].startswith("2024") for e in timeline["events"]))

report = client.get("/api/contradictions").get_json()
subjects = [c["subject"] for c in report["conflicts"]]
check("sources disagreeing on a figure are flagged", len(report["conflicts"]) >= 1, str(subjects))
check("the conflict found is the user count", any("user" in (s or "") for s in subjects))
check("different currencies are never compared",
      not any("revenue" in (s or "").lower() for s in subjects),
      "USD vs EUR must not count as a contradiction")

trace = client.get(f"/api/trace/{first_id}").get_json()
check("a summary can be traced to its source", trace["available"])
strong = [a for a in trace["alignments"] if a["support"] == "strong"]
check("a verbatim sentence is matched", len(strong) >= 1)
if strong:
    span = strong[0]
    check("returned offsets index the source exactly",
          trace["source_text"][span["start"]:span["end"]] == span["matched_text"])


# ---------------------------------------------------------------------------
section("Originals are kept and replayable")
# ---------------------------------------------------------------------------

page = client.get(f"/entry/{second_id}")
check("the entry page links to its stored original", f"/source/{asset_id}".encode() in page.data)
served = client.get(f"/source/{asset_id}")
check("the original is served back byte for byte",
      served.data == b"%PDF-1.4 stand-in for a real document")
check("it is served with its recorded media type", "pdf" in served.headers.get("Content-Type", ""))


# ---------------------------------------------------------------------------
section("Work continues across navigation")
# ---------------------------------------------------------------------------

jobs = client.get("/api/jobs/active").get_json()
check("a running job is visible from any page", len(jobs["active"]) == 1)
check("it reports the step it is on", jobs["active"][0]["latest"] == "Summarising...")
check("the progress bar is present on unrelated pages", b'id="workbar"' in client.get("/analyse").data)


# ---------------------------------------------------------------------------
section("Editing and deleting")
# ---------------------------------------------------------------------------

updated = client.patch(f"/api/entry/{first_id}",
                       json={"title": "Renamed", "notes": "a note", "tags": ["one", "two"]}).get_json()
check("an entry can be renamed", updated["title"] == "Renamed")
check("notes are saved", updated["notes"] == "a note")
check("tags are normalised", set(updated["tags"]) == {"one", "two"})

check("entries can be archived in bulk",
      client.post("/api/entries/bulk", json={"entry_ids": [first_id], "action": "archive"})
      .get_json()["affected"] == 1)

check("an entry can be deleted", client.delete(f"/api/entry/{second_id}").status_code == 200)
with app.app_context():
    from app.models import SourceAsset
    check("deleting also removes its stored file",
          SourceAsset.query.filter_by(entry_id=second_id).count() == 0)


# ---------------------------------------------------------------------------
section("One account cannot reach another's data")
# ---------------------------------------------------------------------------

other = app.test_client()
other.post("/signup",
           data={"email": "someone.else@example.com", "display_name": "Other",
                 "password": "a-long-enough-password", "confirm_password": "a-long-enough-password"},
           follow_redirects=True)
check("another account gets 404, not the entry", other.get(f"/api/entry/{first_id}").status_code == 404)
check("another account cannot trace it", not other.get(f"/api/trace/{first_id}").get_json().get("available"))
check("another account sees an empty library", other.get("/api/canvas").get_json()["nodes"] == [])


# ---------------------------------------------------------------------------
section("The rule language rejects code")
# ---------------------------------------------------------------------------

for hostile in ['__import__("os").system("ls")', 'eval("1+1")', "().__class__.__bases__"]:
    result = client.post("/api/automation/validate", json={"condition": hostile, "actions": []}).get_json()
    check(f"rejected: {hostile[:28]}", not result["valid"])

check("a webhook to a private address is refused",
      not client.post("/api/automation/validate",
                      json={"condition": "word_count > 1",
                            "actions": [{"type": "webhook", "url": "http://localhost:8080/x"}]})
      .get_json()["valid"])


print("\n" + "=" * 62)
print(f"{_passed} passed, {_failed} failed")
print("=" * 62)
sys.exit(1 if _failed else 0)
