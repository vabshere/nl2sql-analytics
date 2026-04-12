from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

TABLE_NAME = "gaming_mental_health"


def _parse_bool_env(key: str, default: bool) -> bool:
    """Parse an env var as a boolean.

    WHY: only "true"/"false" (case-insensitive) are accepted — any other value
    is almost certainly a misconfiguration and should fail loudly rather than
    being silently coerced into a truthy result. Empty string falls back to
    the default so unset-but-declared vars in .env behave like absent vars.
    """
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"Invalid value for {key!r}: {value!r}. Acceptable values: true, false"
    )


def load_descriptions(
    metadata_db_path: Path,
) -> tuple[str, dict[str, dict]]:
    """Read table and column descriptions from the metadata DB.

    Returns (table_description, {col_name: {description, domain}}).
    Returns ("", {}) if the metadata DB does not exist — logs a warning so
    callers degrade gracefully to a no-description DDL.
    """
    if not metadata_db_path.exists():
        logger.warning(
            "Metadata DB not found at %s — DDL will be generated without descriptions.",
            metadata_db_path,
        )
        return "", {}

    # WHY: explicit close — sqlite3 context manager handles transactions only,
    # not connection lifecycle; omitting close causes ResourceWarning on CPython
    conn = sqlite3.connect(metadata_db_path)
    try:
        conn.row_factory = sqlite3.Row

        td_row = conn.execute(
            "SELECT description FROM table_descriptions WHERE table_name = ?",
            (TABLE_NAME,),
        ).fetchone()
        table_description = td_row["description"] if td_row else ""

        # WHY: join avoids a second round-trip and keeps col lookup O(1) via dict
        col_rows = conn.execute(
            """
            SELECT cd.column_name, cd.description, cd.domain
            FROM column_descriptions cd
            JOIN table_descriptions td ON cd.table_description_id = td.id
            WHERE td.table_name = ?
            """,
            (TABLE_NAME,),
        ).fetchall()
    finally:
        conn.close()

    column_descriptions = {
        row["column_name"]: {
            "description": row["description"],
            "domain": row["domain"],
        }
        for row in col_rows
    }

    return table_description, column_descriptions


def introspect_columns(db_path: Path) -> list[tuple[str, str]]:
    """Return (name, type) pairs for all columns in TABLE_NAME via PRAGMA.

    WHY: preserves DB-declared order so callers can reason about determinism.
    """
    # WHY: explicit close — same reason as load_descriptions
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    finally:
        conn.close()
    # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
    columns = [(row[1], row[2]) for row in rows]
    logger.debug("Introspected %d columns from %s", len(columns), db_path)
    return columns


def build_ddl(
    columns: list[tuple[str, str]],
    table_name: str,
    table_description: str,
    column_descriptions: dict[str, dict],
    include_description: bool,
) -> str:
    """Build a CREATE TABLE DDL string.

    When include_description=True:
      - Adds a table-level comment on the first line inside the parentheses
        (only when table_description is non-empty)
      - Adds inline -- comment on each column line where a description exists
      - Columns without a description appear without a comment — never dropped

    When include_description=False:
      - Plain DDL with no comments at all

    Column order follows the `columns` argument exactly.
    The last column line has no trailing comma (required for valid SQLite DDL).
    """
    lines: list[str] = [f"CREATE TABLE {table_name} ("]

    if include_description and table_description:
        lines.append(f"    -- {table_description}")

    for i, (name, col_type) in enumerate(columns):
        is_last = i == len(columns) - 1
        comma = "" if is_last else ","
        base = f"    {name} {col_type}{comma}"

        if include_description and name in column_descriptions:
            desc = column_descriptions[name]["description"]
            line = f"{base}  -- {desc}"
        else:
            if include_description and name not in column_descriptions:
                logger.debug(
                    "No description for column '%s' — emitting without comment.", name
                )
            line = base

        lines.append(line)

    lines.append(");")
    return "\n".join(lines)


def load_schema_context(
    db_path: Path,
    metadata_db_path: Path,
    include_description: bool | None = None,
) -> dict:
    """Orchestrate introspection + description loading + DDL building.

    Returns {"ddl": "<CREATE TABLE ...>"}.

    When include_description is None, reads SCHEMA_INCLUDE_DESCRIPTION env var
    (default true).

    WHY: single-key dict keeps downstream f-string interpolation in generate_sql
    stable — adding future keys (e.g. row_count) does not break existing callers.
    """
    if include_description is None:
        include_description = _parse_bool_env("SCHEMA_INCLUDE_DESCRIPTION", default=True)

    columns = introspect_columns(db_path)
    table_description, column_descriptions = load_descriptions(metadata_db_path)

    ddl = build_ddl(
        columns=columns,
        table_name=TABLE_NAME,
        table_description=table_description,
        column_descriptions=column_descriptions,
        include_description=include_description,
    )

    return {"ddl": ddl}
