"""
Session lifetime: idle timeout, device list, and revocation.

What was wrong before
---------------------
Login called `login_user(user, remember=True)` unconditionally. That
sets a one-year "remember me" cookie on every login, on every machine,
whether or not the person asked for it -- so the app simply never logged
anyone out. On a shared or borrowed computer that isn't convenience,
it's an unlocked door, and the user had no way to see it was open or to
close it remotely.

What happens now
----------------
Three separate mechanisms, because they answer three different
questions:

1. *Absolute lifetime* -- `PERMANENT_SESSION_LIFETIME` caps how long a
   session can live at all, even if used constantly.
2. *Idle timeout* -- a session unused for `IDLE_TIMEOUT_MINUTES` is
   ended on the next request. The clock slides forward on activity, so
   someone working continuously is never interrupted, while a forgotten
   tab expires.
3. *Server-side revocation* -- each login writes a `UserSession` row
   keyed by a random id kept in the signed cookie. Sign-out elsewhere
   flips `revoked`, and the next request from that device is rejected.
   A signed cookie alone can't do this: the cookie is valid until it
   expires and the server has no say. The row is what gives the user
   the say.

"Keep me signed in" remains available, because forcing a re-login every
day on a personal laptop trains people to pick worse passwords. It is
now a deliberate choice with a longer, still-bounded lifetime, not the
silent default.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone

from flask import request, session
from flask_login import current_user, logout_user

from ..extensions import db
from ..models import UserSession

logger = logging.getLogger(__name__)

SESSION_KEY_FIELD = "se_session_key"
LAST_SEEN_FIELD = "se_last_seen"

IDLE_TIMEOUT_MINUTES = 60          # a forgotten tab closes itself after an hour
REMEMBERED_IDLE_DAYS = 30          # "keep me signed in" -- longer, still finite
ABSOLUTE_LIFETIME_DAYS = 90        # nothing lives forever, however active


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes even for timezone=True columns,
    so anything read from the database has to be re-stamped as UTC
    before it can be compared with an aware `now`. Subtracting a naive
    from an aware datetime raises, and it raises inside a before-request
    hook, which takes down every page at once."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def start_session(user_id: int, *, remember: bool) -> str:
    """Record a new logged-in device and return its key."""
    key = secrets.token_urlsafe(32)
    row = UserSession(
        user_id=user_id,
        session_key=key,
        user_agent=(request.headers.get("User-Agent") or "")[:400],
        ip_address=(request.headers.get("X-Forwarded-For", request.remote_addr) or "")[:64],
    )
    db.session.add(row)
    db.session.commit()

    session[SESSION_KEY_FIELD] = key
    session[LAST_SEEN_FIELD] = _now().timestamp()
    session["se_remembered"] = bool(remember)
    session.permanent = True
    return key


def end_current_session() -> None:
    key = session.get(SESSION_KEY_FIELD)
    if key:
        row = UserSession.query.filter_by(session_key=key).first()
        if row is not None:
            row.revoked = True
            db.session.commit()
    session.pop(SESSION_KEY_FIELD, None)
    session.pop(LAST_SEEN_FIELD, None)
    session.pop("se_remembered", None)


def enforce(*, exempt_paths: tuple[str, ...] = ()) -> bool:
    """Called on every request. Returns True if the session was ended.

    Deliberately forgiving about its own failures: if the bookkeeping
    row is missing (an upgrade from before this table existed, say) the
    user stays logged in rather than being thrown out by a schema
    change they had nothing to do with.
    """
    if not current_user.is_authenticated:
        return False
    if request.path.startswith(exempt_paths):
        return False

    key = session.get(SESSION_KEY_FIELD)
    row = UserSession.query.filter_by(session_key=key).first() if key else None

    if row is None:
        # Pre-existing session from before session tracking. Adopt it
        # rather than evicting the user.
        if key is None:
            start_session(current_user.id, remember=bool(session.get("se_remembered", True)))
        return False

    if row.revoked:
        logout_user()
        end_current_session()
        return True

    now = _now()
    created = _as_aware(row.created_at) or now
    if now - created > timedelta(days=ABSOLUTE_LIFETIME_DAYS):
        row.revoked = True
        db.session.commit()
        logout_user()
        end_current_session()
        return True

    idle_limit = timedelta(days=REMEMBERED_IDLE_DAYS) if session.get("se_remembered") else timedelta(minutes=IDLE_TIMEOUT_MINUTES)
    last_seen = _as_aware(row.last_seen_at) or created
    if now - last_seen > idle_limit:
        row.revoked = True
        db.session.commit()
        logout_user()
        end_current_session()
        return True

    # Slide the window forward, but only write once a minute. Without
    # this every asset request would be a database write, which on a
    # page with twenty requests means twenty pointless commits.
    if (now - last_seen) > timedelta(seconds=60):
        row.last_seen_at = now
        db.session.commit()
    session[LAST_SEEN_FIELD] = now.timestamp()
    return False


def seconds_remaining() -> int | None:
    """How long until this session idles out, for the countdown warning.
    None when the session is remembered and the answer is 'not soon'."""
    if not current_user.is_authenticated or session.get("se_remembered"):
        return None
    key = session.get(SESSION_KEY_FIELD)
    row = UserSession.query.filter_by(session_key=key).first() if key else None
    if row is None:
        return None
    last_seen = _as_aware(row.last_seen_at) or _now()
    remaining = timedelta(minutes=IDLE_TIMEOUT_MINUTES) - (_now() - last_seen)
    return max(0, int(remaining.total_seconds()))


def list_sessions(user_id: int) -> list[dict]:
    rows = (
        UserSession.query.filter_by(user_id=user_id, revoked=False)
        .order_by(UserSession.last_seen_at.desc())
        .all()
    )
    current_key = session.get(SESSION_KEY_FIELD)
    return [
        {
            "id": r.id,
            "current": r.session_key == current_key,
            "device": _describe_agent(r.user_agent or ""),
            "user_agent": r.user_agent or "",
            "ip_address": r.ip_address or "",
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else "",
        }
        for r in rows
    ]


def revoke_session(user_id: int, session_id: int) -> bool:
    row = UserSession.query.filter_by(id=session_id, user_id=user_id).first()
    if row is None:
        return False
    row.revoked = True
    db.session.commit()
    return True


def revoke_all_other_sessions(user_id: int) -> int:
    current_key = session.get(SESSION_KEY_FIELD)
    rows = UserSession.query.filter_by(user_id=user_id, revoked=False).all()
    count = 0
    for row in rows:
        if row.session_key != current_key:
            row.revoked = True
            count += 1
    db.session.commit()
    return count


def _describe_agent(agent: str) -> str:
    """A readable device name. User-agent strings are famously unreliable
    -- every browser claims to be several others -- so this is a best
    effort for recognising your own devices in a list, nothing more."""
    agent_lower = agent.lower()
    if "iphone" in agent_lower:
        return "iPhone"
    if "ipad" in agent_lower:
        return "iPad"
    if "android" in agent_lower:
        return "Android device"
    system = "Mac" if "mac os" in agent_lower or "macintosh" in agent_lower else (
        "Windows" if "windows" in agent_lower else ("Linux" if "linux" in agent_lower else "Unknown system")
    )
    browser = (
        "Edge" if "edg/" in agent_lower else
        "Chrome" if "chrome" in agent_lower and "safari" in agent_lower else
        "Firefox" if "firefox" in agent_lower else
        "Safari" if "safari" in agent_lower else "Browser"
    )
    return f"{browser} on {system}"


def purge_expired(days: int = 120) -> int:
    """Housekeeping so the table doesn't grow without bound."""
    cutoff = _now() - timedelta(days=days)
    deleted = UserSession.query.filter(UserSession.last_seen_at < cutoff).delete()
    db.session.commit()
    return deleted
