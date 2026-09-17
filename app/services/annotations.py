"""
Personal annotations -- the one feature in this app that is deliberately
NOT AI. Every other feature runs something through Gemini; this is just
you, highlighting a passage from something you've read and writing your
own note on it, saved and attached permanently to that knowledge-base
entry. It exists to prove the app isn't "all API calls and no substance"
-- your own curation is first-class data here, not an afterthought.
"""
from __future__ import annotations

from ..extensions import db
from ..models import Annotation


class AnnotationError(RuntimeError):
    pass


def _annotation_dict(a: Annotation) -> dict:
    return {
        "id": a.id,
        "entry_id": a.entry_id,
        "quote": a.quote,
        "note": a.note,
        "created_at": a.created_at.isoformat(),
    }


def add_annotation(user_id: int, entry_id: int, quote: str, note: str) -> int:
    note = (note or "").strip()
    if not note:
        raise AnnotationError("Write a note before saving.")
    annotation = Annotation(user_id=user_id, entry_id=entry_id, quote=(quote or "").strip(), note=note)
    db.session.add(annotation)
    db.session.commit()
    return annotation.id


def list_annotations_for(user_id: int, entry_id: int) -> list[dict]:
    annotations = (
        Annotation.query.filter_by(user_id=user_id, entry_id=entry_id).order_by(Annotation.id.desc()).all()
    )
    return [_annotation_dict(a) for a in annotations]


def delete_annotation(annotation_id: int, user_id: int) -> None:
    Annotation.query.filter_by(id=annotation_id, user_id=user_id).delete()
    db.session.commit()


def annotation_count(user_id: int) -> int:
    return Annotation.query.filter_by(user_id=user_id).count()
