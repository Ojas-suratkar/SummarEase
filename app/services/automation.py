"""
Automation: rules the user writes that run without them watching.

"When a watched page changes and it mentions a funding round, pull out
the numbers, tag it, and email me." That sentence is the feature. It
turns the app from a place you visit into something that works while
you're asleep.

How the safety works
--------------------
A rule's condition is text the user typed, stored in a database, and
evaluated later by the server. If we passed that to `eval()` we would
have built a remote code execution vulnerability with a friendly UI on
top. So conditions are parsed and evaluated by our own small
interpreter in app/core/rules.py, which has no access to Python objects,
no attribute access, no imports, and a step and time budget.

Actions are *data*, never code. `rules.evaluate_rule` returns a list of
things to do; this module is the only place that decides what those
things mean and performs them. That separation is what keeps a rule
from doing anything the product doesn't already offer as a feature.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..core import quantities as quantities_core
from ..core import rules as rules_core
from ..extensions import db
from ..models import AutomationRule, HistoryEntry, RuleRun

logger = logging.getLogger(__name__)

TRIGGERS = [
    ("entry.created", "A new summary is saved"),
    ("entry.updated", "A summary is edited"),
    ("watch.changed", "A watched page changes"),
    ("digest.weekly", "The weekly digest runs"),
    ("manual", "Only when I run it by hand"),
]

ACTION_TYPES = [
    ("tag", "Add a tag"),
    ("archive", "Archive it"),
    ("pin", "Pin it"),
    ("note", "Append a note"),
    ("extract_numbers", "Pull out the numbers"),
    ("add_to_timeline", "Add its dates to the timeline"),
    ("webhook", "Send it to a webhook"),
    ("email", "Email me"),
]

_VALID_ACTIONS = {a for a, _ in ACTION_TYPES}


# ---------------------------------------------------------------------------
# Rule storage
# ---------------------------------------------------------------------------


def _rule_to_core(row: AutomationRule) -> rules_core.Rule:
    try:
        actions = json.loads(row.actions or "[]")
    except json.JSONDecodeError:
        actions = []
    return rules_core.Rule(
        id=str(row.id),
        name=row.name,
        trigger=row.trigger,
        condition=row.condition or "",
        actions=actions,
        enabled=bool(row.enabled),
    )


def rule_dict(row: AutomationRule) -> dict:
    try:
        actions = json.loads(row.actions or "[]")
    except json.JSONDecodeError:
        actions = []
    description = ""
    if row.condition:
        try:
            description = rules_core.describe(row.condition)
        except Exception:
            description = row.condition
    return {
        "id": row.id,
        "name": row.name,
        "trigger": row.trigger,
        "condition": row.condition or "",
        "description": description,
        "actions": actions,
        "enabled": bool(row.enabled),
        "run_count": row.run_count,
        "match_count": row.match_count,
        "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
        "last_error": row.last_error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def list_rules(user_id: int) -> list[dict]:
    rows = AutomationRule.query.filter_by(user_id=user_id).order_by(AutomationRule.id.desc()).all()
    return [rule_dict(r) for r in rows]


def validate_rule(condition: str, actions: list[dict]) -> dict:
    """Check a rule before it is saved, so mistakes surface while the
    user is still looking at the form rather than silently at 3am."""
    result = {"valid": True, "error": None, "description": "", "identifiers": []}

    condition = (condition or "").strip()
    if condition:
        check = rules_core.validate(condition)
        if not check.get("valid"):
            return {"valid": False, "error": check.get("error"), "description": "", "identifiers": []}
        result["identifiers"] = check.get("identifiers", [])
        try:
            result["description"] = rules_core.describe(condition)
        except Exception:
            result["description"] = condition
    else:
        result["description"] = "every time (no condition set)"

    for action in actions or []:
        kind = (action or {}).get("type")
        if kind not in _VALID_ACTIONS:
            return {"valid": False, "error": f"Unknown action: {kind!r}", "description": "", "identifiers": []}
        if kind == "webhook":
            url = (action.get("url") or "").strip()
            # Only outbound HTTPS to a real host. A webhook pointing at
            # localhost or a private range would turn this feature into
            # a server-side request forgery primitive.
            if not url.startswith("https://"):
                return {"valid": False, "error": "Webhook URLs must start with https://", "description": "", "identifiers": []}
            if any(bad in url for bad in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.", "[::1]")):
                return {"valid": False, "error": "Webhook URLs can't point at internal addresses.", "description": "", "identifiers": []}
        if kind == "tag" and not (action.get("value") or "").strip():
            return {"valid": False, "error": "A tag action needs a tag to add.", "description": "", "identifiers": []}

    return result


def save_rule(user_id: int, *, name: str, trigger: str, condition: str, actions: list[dict], enabled: bool = True, rule_id: int | None = None) -> dict:
    check = validate_rule(condition, actions)
    if not check["valid"]:
        raise ValueError(check["error"])

    if rule_id:
        row = AutomationRule.query.filter_by(id=rule_id, user_id=user_id).first()
        if row is None:
            raise ValueError("That rule doesn't exist.")
    else:
        row = AutomationRule(user_id=user_id)
        db.session.add(row)

    row.name = (name or "Untitled rule").strip()[:160]
    row.trigger = trigger if trigger in {t for t, _ in TRIGGERS} else "entry.created"
    row.condition = (condition or "").strip()
    row.actions = json.dumps(actions or [])
    row.enabled = bool(enabled)
    db.session.commit()
    return rule_dict(row)


def delete_rule(user_id: int, rule_id: int) -> bool:
    row = AutomationRule.query.filter_by(id=rule_id, user_id=user_id).first()
    if row is None:
        return False
    RuleRun.query.filter_by(rule_id=rule_id, user_id=user_id).delete()
    db.session.delete(row)
    db.session.commit()
    return True


def toggle_rule(user_id: int, rule_id: int, enabled: bool) -> bool:
    row = AutomationRule.query.filter_by(id=rule_id, user_id=user_id).first()
    if row is None:
        return False
    row.enabled = bool(enabled)
    db.session.commit()
    return True


# ---------------------------------------------------------------------------
# Running rules
# ---------------------------------------------------------------------------


def context_for_entry(entry: HistoryEntry) -> dict:
    """The variables a rule can see when it runs against an entry.

    This is an allow-list by construction: the interpreter can only read
    what appears in this dict, so there is nothing to escape into.
    """
    text = (entry.source_text or "") or (entry.summary or "")
    source_ref = entry.source_ref or ""
    domain = ""
    if "://" in source_ref:
        domain = source_ref.split("://", 1)[1].split("/", 1)[0].lower()

    return {
        "source_type": entry.source_type or "",
        "kind": entry.source_type or "",
        "title": entry.display_title,
        "text": text,
        "summary": entry.summary or "",
        "notes": entry.notes or "",
        "tags": entry.tag_list,
        "domain": domain,
        "url": source_ref if "://" in source_ref else "",
        "word_count": len(text.split()),
        "archived": bool(entry.is_archived),
        "pinned": bool(entry.is_pinned),
        "created_at": entry.created_at.isoformat() if entry.created_at else "",
    }


def run_for_entry(user_id: int, entry_id: int, *, trigger: str = "entry.created", dry_run: bool = False) -> list[dict]:
    entry = HistoryEntry.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry is None:
        return []

    rows = AutomationRule.query.filter_by(user_id=user_id, enabled=True).all()
    applicable = [r for r in rows if r.trigger == trigger or trigger == "manual"]
    if not applicable:
        return []

    context = context_for_entry(entry)
    outcomes = []

    for row in applicable:
        # "Run it now" means "pretend this rule's own trigger just fired",
        # otherwise the rule's trigger gate rejects the manual event and
        # every hand-run reports no match -- which looks like a broken
        # condition when the condition is fine.
        event_type = row.trigger if trigger == "manual" else trigger
        event = {"type": event_type, "entry_id": entry_id, "manual": trigger == "manual"}
        try:
            outcome = rules_core.evaluate_rule(_rule_to_core(row), event, context)
        except Exception as exc:  # pragma: no cover
            outcome = {"matched": False, "actions": [], "error": str(exc), "explain": ""}

        row.run_count = (row.run_count or 0) + 1
        row.last_run_at = datetime.now(timezone.utc)
        row.last_error = outcome.get("error")

        performed = []
        if outcome.get("matched"):
            row.match_count = (row.match_count or 0) + 1
            if not dry_run:
                performed = _perform_actions(user_id, entry, outcome.get("actions", []))
            else:
                performed = [{"type": a.get("type"), "status": "would run"} for a in outcome.get("actions", [])]

        db.session.add(
            RuleRun(
                user_id=user_id,
                rule_id=row.id,
                matched=bool(outcome.get("matched")),
                explain=outcome.get("explain", ""),
                actions_taken=json.dumps(performed),
                error=outcome.get("error"),
            )
        )
        outcomes.append(
            {
                "rule_id": row.id,
                "rule_name": row.name,
                "matched": bool(outcome.get("matched")),
                "explain": outcome.get("explain", ""),
                "actions": performed,
                "error": outcome.get("error"),
            }
        )

    db.session.commit()
    return outcomes


def _perform_actions(user_id: int, entry: HistoryEntry, actions: list[dict]) -> list[dict]:
    """Carry out the actions a matched rule asked for.

    Every branch is a capability the product already has. There is no
    generic "run this" action and there never should be.
    """
    performed = []

    for action in actions or []:
        kind = (action or {}).get("type")
        try:
            if kind == "tag":
                tag = (action.get("value") or "").strip().lower()
                if tag and tag not in entry.tag_list:
                    existing = entry.tag_list + [tag]
                    entry.tags = ",".join(existing)
                performed.append({"type": kind, "status": "tagged", "value": tag})

            elif kind == "archive":
                entry.is_archived = True
                performed.append({"type": kind, "status": "archived"})

            elif kind == "pin":
                entry.is_pinned = True
                performed.append({"type": kind, "status": "pinned"})

            elif kind == "note":
                note = (action.get("value") or "").strip()
                if note:
                    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    entry.notes = ((entry.notes or "") + f"\n[{stamp}, by rule] {note}").strip()
                performed.append({"type": kind, "status": "noted"})

            elif kind == "extract_numbers":
                text = (entry.source_text or "") or (entry.summary or "")
                found = quantities_core.extract_quantities(text[:200_000])
                performed.append({"type": kind, "status": "extracted", "count": len(found)})

            elif kind == "add_to_timeline":
                # Timeline events are derived on read from the corpus, so
                # an entry being present is all that's required. Recorded
                # here so the audit trail shows the rule considered it.
                performed.append({"type": kind, "status": "included"})

            elif kind == "webhook":
                performed.append(_send_webhook(action.get("url", ""), entry))

            elif kind == "email":
                # Email needs SMTP credentials this deployment doesn't
                # have yet. Rather than silently doing nothing, the run
                # is recorded as queued so the user can see it was
                # matched and why nothing arrived.
                performed.append({"type": kind, "status": "queued (no mail server configured)"})

            else:
                performed.append({"type": kind, "status": "skipped (unknown action)"})

        except Exception as exc:  # pragma: no cover
            performed.append({"type": kind, "status": f"failed: {exc}"})

    db.session.commit()
    return performed


def _send_webhook(url: str, entry: HistoryEntry) -> dict:
    """POST a compact payload. Short timeout, no redirects followed, and
    failures are recorded rather than raised -- one unreachable endpoint
    must not stop the other rules in the batch."""
    if not url.startswith("https://"):
        return {"type": "webhook", "status": "refused (not https)"}
    try:
        import requests

        response = requests.post(
            url,
            json={
                "event": "summarease.rule",
                "entry_id": entry.id,
                "title": entry.display_title,
                "source_type": entry.source_type,
                "summary": (entry.summary or "")[:2000],
                "tags": entry.tag_list,
            },
            timeout=6,
            allow_redirects=False,
        )
        return {"type": "webhook", "status": f"sent ({response.status_code})"}
    except Exception as exc:
        return {"type": "webhook", "status": f"failed: {type(exc).__name__}"}


def recent_runs(user_id: int, limit: int = 50) -> list[dict]:
    rows = (
        RuleRun.query.filter_by(user_id=user_id)
        .order_by(RuleRun.id.desc())
        .limit(limit)
        .all()
    )
    names = {r.id: r.name for r in AutomationRule.query.filter_by(user_id=user_id).all()}
    out = []
    for row in rows:
        try:
            actions = json.loads(row.actions_taken or "[]")
        except json.JSONDecodeError:
            actions = []
        out.append(
            {
                "id": row.id,
                "rule_id": row.rule_id,
                "rule_name": names.get(row.rule_id, "(deleted rule)"),
                "matched": row.matched,
                "explain": row.explain,
                "actions": actions,
                "error": row.error,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
        )
    return out


def example_rules() -> list[dict]:
    """Starting points. An empty rule builder with a blank expression box
    is intimidating; three working examples make the language obvious."""
    return [
        {
            "name": "Flag long research reads",
            "trigger": "entry.created",
            "condition": 'word_count > 2000 and source_type == "pdf"',
            "actions": [{"type": "tag", "value": "long-read"}],
        },
        {
            "name": "Watch for funding news",
            "trigger": "watch.changed",
            "condition": 'text mentions "funding" or text mentions "raised"',
            "actions": [{"type": "tag", "value": "funding"}, {"type": "extract_numbers"}],
        },
        {
            "name": "Archive short clips automatically",
            "trigger": "entry.created",
            "condition": 'kind in ["audio", "video"] and word_count < 150',
            "actions": [{"type": "archive"}],
        },
    ]
