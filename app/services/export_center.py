"""
Export center -- your data isn't trapped here.

Everything else in this app assumes you're staying inside it. This is the
opposite kind of feature: it gets your data *out*, into formats real
tools already understand, so summarizing something here doesn't mean
re-typing it somewhere else later.

- **Anki deck (.apkg)** -- every due-for-review flashcard
  (spaced_repetition.py), as a real Anki package built with `genanki`
  (the standard library for this -- an .apkg is a zipped SQLite database
  with a specific schema, not something worth hand-rolling).
- **Obsidian-compatible Markdown vault (.zip)** -- every knowledge-base
  entry as its own Markdown file with YAML frontmatter, and the
  auto-linked knowledge graph (knowledge_graph.py) rendered as real
  Obsidian `[[wikilinks]]` between notes -- so relationships Gemini found
  automatically show up as an actual Obsidian graph, not just inside this
  app.
- **PDF booklet** -- the whole knowledge base (or one entry) as a plain,
  readable PDF, built with `fpdf2`.
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone

from . import history_store, spaced_repetition

_ANKI_MODEL_ID = 1607392319
_ANKI_DECK_ID = 2059400110


class ExportError(RuntimeError):
    pass


def _safe_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\- ]", "", text or "").strip() or "untitled"
    return text[:max_len].replace(" ", "-")


def build_anki_deck(user_id: int) -> bytes:
    """All flashcards (services/spaced_repetition.py), regardless of due
    date, as a real .apkg file you can double-click into Anki."""
    import genanki

    model = genanki.Model(
        _ANKI_MODEL_ID,
        "SummarEase Basic",
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Question}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Answer}}',
            }
        ],
    )
    deck = genanki.Deck(_ANKI_DECK_ID, "SummarEase")

    cards = spaced_repetition.due_flashcards(user_id, limit=100000)
    if not cards:
        raise ExportError("No flashcards to export yet -- generate some from a result first.")

    for card in cards:
        deck.add_note(genanki.Note(model=model, fields=[card["question"], card["answer"]]))

    package = genanki.Package(deck)
    buffer = io.BytesIO()
    package.write_to_file(buffer)  # genanki writes to a path or file-like object
    return buffer.getvalue()


def _entry_to_markdown(entry: dict, links: list[dict]) -> str:
    created = (entry.get("created_at") or "")[:10]
    frontmatter = (
        "---\n"
        f"source_type: {entry.get('source_type', '')}\n"
        f"source_ref: \"{(entry.get('source_ref') or '').replace(chr(34), chr(39))}\"\n"
        f"created: {created}\n"
        "---\n\n"
    )
    body = f"# {entry.get('source_ref') or entry.get('source_type', 'Untitled')}\n\n{entry.get('summary', '')}\n"
    if links:
        body += "\n## Related\n\n"
        for link in links:
            title = _safe_filename(link.get("source_ref") or f"entry-{link['id']}")
            body += f"- {link['relationship']}: [[{title}]]"
            if link.get("rationale"):
                body += f" -- {link['rationale']}"
            body += "\n"
    return frontmatter + body


def build_obsidian_vault(user_id: int) -> bytes:
    """Every knowledge-base entry as its own Markdown note, zipped into a
    folder Obsidian can open directly as a vault -- auto-linked entries
    (knowledge_graph.py) become real [[wikilinks]] between notes."""
    entries = history_store.list_recent(user_id, limit=100000)
    if not entries:
        raise ExportError("Your knowledge base is empty -- summarize a few things first.")

    buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            links = history_store.get_links_for(user_id, entry["id"])
            markdown = _entry_to_markdown(entry, links)
            base_name = _safe_filename(entry.get("source_ref") or f"entry-{entry['id']}")
            name = base_name
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base_name}-{suffix}"
            used_names.add(name)
            zf.writestr(f"SummarEase Vault/{name}.md", markdown)
    return buffer.getvalue()


def build_knowledge_base_pdf(user_id: int) -> bytes:
    """The whole knowledge base as one readable PDF booklet."""
    from fpdf import FPDF

    entries = history_store.list_recent(user_id, limit=100000)
    if not entries:
        raise ExportError("Your knowledge base is empty -- summarize a few things first.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "SummarEase Knowledge Base", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 8, f"Exported {datetime.now(timezone.utc).strftime('%Y-%m-%d')} -- {len(entries)} entries", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for entry in entries:
        pdf.set_font("Helvetica", "B", 13)
        title = (entry.get("source_ref") or entry.get("source_type") or "Untitled").encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 8, title)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(110, 110, 110)
        meta = f"{entry.get('source_type', '')} -- {(entry.get('created_at') or '')[:10]}"
        pdf.cell(0, 6, meta, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 11)
        body = (entry.get("summary") or "").encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, body)
        pdf.ln(4)

    return bytes(pdf.output())
