"""Apply the SQL files in `migrations/` in name order, each exactly once.

Each file runs in its own transaction together with its row in `schema_migrations`,
so a failing migration leaves no half-applied schema behind.
"""

from __future__ import annotations

from importlib.resources import files

import psycopg

# Any constant works: it only has to be the same for every process that migrates.
_LOCK_ID = 7_014_412


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations and return their names."""
    # Two processes starting together (app and worker) must not both apply a file.
    conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_ID,))
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version    text PRIMARY KEY,
                   applied_at timestamptz NOT NULL DEFAULT now()
               )"""
        )
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        migrations = files("arxiv_atlas.db").joinpath("migrations")
        sql_files = sorted(
            (f for f in migrations.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name
        )
        newly_applied = []
        for sql_file in sql_files:
            version = sql_file.name.removesuffix(".sql")
            if version in applied:
                continue
            with conn.transaction():
                # Bytes, not str: psycopg types queries as LiteralString to block SQL
                # built from runtime strings, and a file we ship is trusted but not a literal.
                conn.execute(sql_file.read_bytes())
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            newly_applied.append(version)
        return newly_applied
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
