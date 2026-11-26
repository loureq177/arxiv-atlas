"""Shared fixtures. Database tests run against a real PostgreSQL with pgvector.

Set e.g. `TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5432/atlas_test`.
The fixture drops the whole schema, so never point it at your real database.
"""

import os

import psycopg
import pytest

from arxiv_atlas.db.migrate import migrate

TEST_DSN = os.environ.get("TEST_DATABASE_URL", "")


@pytest.fixture
def db():
    """A connection to an empty, fully migrated database."""
    if not TEST_DSN:
        pytest.skip("TEST_DATABASE_URL not set")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        migrate(conn)
        yield conn
