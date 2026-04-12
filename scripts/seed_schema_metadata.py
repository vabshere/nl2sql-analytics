"""
Schema Metadata Seeder

Usage:
    python scripts/seed_schema_metadata.py [--json PATH] [--db PATH]

What it does:
- Reads table and column descriptions from a JSON file
- Creates (or replaces) data/schema_metadata.sqlite with two tables:
    table_descriptions  — one row per table
    column_descriptions — one row per column with FK to table_descriptions
- UUIDs are generated in Python and stored as TEXT (SQLite has no native UUID type)

Default paths:
- JSON: data/schema_metadata.json
- DB:   data/schema_metadata.sqlite
"""

import argparse
import json
import sqlite3
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "schema_metadata.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "schema_metadata.sqlite"

CREATE_TABLE_DESCRIPTIONS = """
CREATE TABLE table_descriptions (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL
)
"""

CREATE_COLUMN_DESCRIPTIONS = """
CREATE TABLE column_descriptions (
    id TEXT PRIMARY KEY,
    table_description_id TEXT NOT NULL REFERENCES table_descriptions(id),
    column_name TEXT NOT NULL,
    domain TEXT,
    description TEXT NOT NULL,
    UNIQUE (table_description_id, column_name)
)
"""


def seed_from_json(json_path: Path, db_path: Path) -> None:
    """Drop, recreate, and populate metadata tables from json_path into db_path."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON metadata file not found: {json_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    tables = metadata.get("tables", {})

    with sqlite3.connect(db_path) as conn:
        # WHY: replace strategy — drop and recreate for a clean slate on every run
        conn.execute("DROP TABLE IF EXISTS column_descriptions")
        conn.execute("DROP TABLE IF EXISTS table_descriptions")
        conn.execute(CREATE_TABLE_DESCRIPTIONS)
        conn.execute(CREATE_COLUMN_DESCRIPTIONS)

        for table_name, table_data in tables.items():
            table_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO table_descriptions (id, table_name, description) VALUES (?, ?, ?)",
                (table_id, table_name, table_data.get("description", "")),
            )

            for column_name, column_data in table_data.get("columns", {}).items():
                conn.execute(
                    """
                    INSERT INTO column_descriptions
                        (id, table_description_id, column_name, domain, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        table_id,
                        column_name,
                        column_data.get("domain"),
                        column_data.get("description", ""),
                    ),
                )

        conn.commit()

    print(f"Seeded {len(tables)} table(s) into {db_path}")
    for table_name, table_data in tables.items():
        col_count = len(table_data.get("columns", {}))
        print(f"  {table_name}: {col_count} column(s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed schema metadata SQLite DB from JSON."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to JSON metadata file (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to output SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    try:
        seed_from_json(args.json, args.db)
        print("\n✓ Schema metadata seeded successfully.")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
