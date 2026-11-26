"""PostgreSQL persistence for paper embeddings and nearest-neighbour search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from arxiv_atlas.embeddings.model import Vectors

# The text a paper is embedded from is defined once, here, and hashed in SQL, so a
# changed title or abstract is detected without a second definition in Python.
_PENDING = """
FROM papers p
CROSS JOIN LATERAL (SELECT p.title || E'\\n\\n' || p.abstract AS text) d
LEFT JOIN paper_embeddings e ON e.arxiv_id = p.arxiv_id
WHERE NOT p.is_deleted
  AND (%(since)s::date IS NULL OR p.published_at >= %(since)s::date)
  AND (e.arxiv_id IS NULL OR e.model <> %(model)s OR e.text_md5 <> md5(d.text))
"""


@dataclass(frozen=True, slots=True)
class PendingPaper:
    arxiv_id: str
    text: str
    text_md5: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    arxiv_id: str
    title: str
    primary_category: str
    published_at: datetime
    similarity: float


class EmbeddingStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        conn.autocommit = True
        register_vector(conn)  # needs the extension, so run migrations first
        self._conn = conn

    def count_pending(self, model: str, since: date | None) -> int:
        row = self._conn.execute(
            "SELECT count(*) " + _PENDING, {"model": model, "since": since}
        ).fetchone()
        return int(row[0]) if row else 0

    def pending(self, model: str, since: date | None, after: str, limit: int) -> list[PendingPaper]:
        """Papers with no up-to-date embedding, `limit` at a time, ordered by id.

        `after` is the last id of the previous batch (keyset pagination): each call
        resumes where the last one stopped instead of rescanning from the start.
        """
        rows = self._conn.execute(
            "SELECT p.arxiv_id, d.text, md5(d.text) "
            + _PENDING
            + " AND p.arxiv_id > %(after)s ORDER BY p.arxiv_id LIMIT %(limit)s",
            {"model": model, "since": since, "after": after, "limit": limit},
        ).fetchall()
        return [PendingPaper(*row) for row in rows]

    def save(self, model: str, papers: list[PendingPaper], vectors: Vectors) -> None:
        if len(papers) != len(vectors):
            raise ValueError(f"{len(papers)} papers but {len(vectors)} vectors")
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO paper_embeddings (arxiv_id, model, text_md5, embedding)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (arxiv_id) DO UPDATE SET
                       model = EXCLUDED.model, text_md5 = EXCLUDED.text_md5,
                       embedding = EXCLUDED.embedding, created_at = now()""",
                [
                    (p.arxiv_id, model, p.text_md5, HalfVector(v))
                    for p, v in zip(papers, vectors, strict=True)
                ],
            )

    def search(self, model: str, query: Vectors, k: int) -> list[SearchHit]:
        """The k papers closest to `query`, most similar first."""
        rows = self._conn.execute(
            """SELECT p.arxiv_id, p.title, p.primary_category, p.published_at,
                      1 - (e.embedding <=> %(q)s) AS similarity
               FROM paper_embeddings e
               JOIN papers p ON p.arxiv_id = e.arxiv_id
               WHERE e.model = %(model)s AND NOT p.is_deleted
               ORDER BY e.embedding <=> %(q)s
               LIMIT %(k)s""",
            {"q": HalfVector(query), "model": model, "k": k},
        ).fetchall()
        return [SearchHit(*row) for row in rows]
