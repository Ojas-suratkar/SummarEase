"""
Additive schema migrations, run automatically at startup.

Why this file exists
--------------------
`db.create_all()` creates tables that don't exist yet. It does *not*
touch tables that already exist, so a column added to a model after the
database was first created is silently missing at runtime -- the app
starts fine and then throws `no such column` on the first query that
touches it. That is exactly the kind of failure that loses a user's
trust, because it looks like the app corrupted their data when in fact
the data is fine and the schema simply drifted.

There's a real database already in use here with real summaries in it,
so "delete the file and start again" is not an acceptable migration
strategy. Instead, on every boot we compare the live schema against
what the models declare and issue `ALTER TABLE ... ADD COLUMN` for
anything missing.

Deliberate limits
-----------------
This only ever *adds*. It never drops a column, never renames one, and
never rewrites data. Those operations are destructive and ambiguous, and
doing them automatically at startup -- unattended, with no backup step
and no human watching -- is how people lose work. If a genuinely
destructive change is ever needed, it should be a deliberate, reviewed,
one-off script, not something that happens because someone restarted
the server.

Every statement is idempotent: run it twice and the second run is a
no-op, because we check the live column list first. Works on SQLite
(local) and Postgres (deployed); the small dialect differences are
handled in `_column_sql`.
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from .extensions import db

logger = logging.getLogger(__name__)


# (table, column, SQL type, default clause) -- kept as data rather than
# hand-written ALTER statements so adding a column here is one line and
# can't accidentally be written in a dialect-specific way.
_ADDITIVE_COLUMNS: list[tuple[str, str, str, str]] = [
    ("history_entries", "title", "VARCHAR(300)", ""),
    ("history_entries", "notes", "TEXT", ""),
    ("history_entries", "tags", "VARCHAR(500)", "DEFAULT ''"),
    ("history_entries", "detail_level", "VARCHAR(20)", "DEFAULT 'standard'"),
    ("history_entries", "is_archived", "BOOLEAN", "DEFAULT 0"),
    ("history_entries", "is_pinned", "BOOLEAN", "DEFAULT 0"),
    ("history_entries", "updated_at", "TIMESTAMP", ""),
]


def _boolean_default(dialect_name: str, clause: str) -> str:
    """Postgres rejects `DEFAULT 0` on a BOOLEAN column -- it wants
    `DEFAULT false`. SQLite stores booleans as integers and accepts 0.
    Same model definition, two spellings."""
    if dialect_name == "postgresql":
        return clause.replace("DEFAULT 0", "DEFAULT false").replace("DEFAULT 1", "DEFAULT true")
    return clause


def _column_sql(dialect_name: str, table: str, column: str, sql_type: str, default: str) -> str:
    default = _boolean_default(dialect_name, default)
    return f"ALTER TABLE {table} ADD COLUMN {column} {sql_type} {default}".strip()


def run_migrations() -> list[str]:
    """Bring the live schema up to date with the models. Returns the list
    of statements actually applied, so startup can log what changed (and
    log nothing on the overwhelmingly common no-op boot)."""
    applied: list[str] = []
    inspector = inspect(db.engine)
    dialect_name = db.engine.dialect.name

    try:
        existing_tables = set(inspector.get_table_names())
    except Exception:  # pragma: no cover -- a brand new database
        return applied

    for table, column, sql_type, default in _ADDITIVE_COLUMNS:
        if table not in existing_tables:
            continue  # create_all() will build it complete; nothing to patch
        try:
            columns = {c["name"] for c in inspector.get_columns(table)}
        except Exception:
            continue
        if column in columns:
            continue

        statement = _column_sql(dialect_name, table, column, sql_type, default)
        try:
            with db.engine.begin() as connection:
                connection.execute(text(statement))
            applied.append(statement)
            logger.info("Schema migration applied: %s", statement)
        except Exception as exc:  # pragma: no cover
            # A failure here must not stop the app from booting: the
            # missing column degrades one feature, a failed boot takes
            # everything down. Log loudly and continue.
            logger.warning("Schema migration failed (%s): %s", statement, exc)

    _backfill_titles()
    return applied


def _backfill_titles() -> None:
    """Give existing entries a sensible title instead of leaving the
    column null. Runs once -- after the first pass every row has a
    title, so the UPDATE matches nothing and costs a single index scan."""
    try:
        with db.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE history_entries SET title = source_ref "
                    "WHERE (title IS NULL OR title = '') AND source_ref IS NOT NULL AND source_ref != ''"
                )
            )
    except Exception as exc:  # pragma: no cover
        logger.debug("Title backfill skipped: %s", exc)
