"""
Shareable public links.

Everything else in this app requires being logged in as the account that
owns the data. This is the one feature that produces a URL someone
*without* an account here can open -- a plain, read-only page with just
that one summary on it, no login, no way to browse anything else in your
knowledge base. The token is an unguessable random string (`secrets`,
not a sequential ID), so knowing one share link tells you nothing about
any other entry's link.
"""
from __future__ import annotations

import secrets

from . import history_store
from ..extensions import db
from ..models import Share


class SharingError(RuntimeError):
    pass


def create_share(user_id: int, entry_id: int) -> str:
    if history_store.get_entry_for_user(entry_id, user_id) is None:
        raise SharingError("That entry no longer exists.")
    existing = Share.query.filter_by(entry_id=entry_id, user_id=user_id).first()
    if existing:
        return existing.token
    token = secrets.token_urlsafe(16)
    db.session.add(Share(token=token, user_id=user_id, entry_id=entry_id))
    db.session.commit()
    return token


def resolve_share(token: str) -> dict | None:
    """Deliberately unscoped by user -- this is the one lookup in the
    app meant to work for someone who isn't logged in at all."""
    share = Share.query.filter_by(token=token).first()
    if share is None:
        return None
    return history_store.get_entry(share.entry_id)


def revoke_share(user_id: int, entry_id: int) -> None:
    Share.query.filter_by(entry_id=entry_id, user_id=user_id).delete()
    db.session.commit()
