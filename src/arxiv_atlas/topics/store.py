"""PostgreSQL persistence for topic runs, topics and each paper's topic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import LiteralString

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from arxiv_atlas.embeddings.model import Vectors
from arxiv_atlas.topics.assign import Reference
from arxiv_atlas.topics.fit import FitResult

# Papers a run covers: vectors from the run's model, published on or after its date.
_IN_SCOPE = """
FROM paper_embeddings e
JOIN papers p ON p.arxiv_id = e.arxiv_id
WHERE e.model = %(model)s AND NOT p.is_deleted AND p.published_at >= %(since)s
"""


@dataclass(frozen=True, slots=True)
class TopicRun:
    id: int
    model: str
    published_since: date


@dataclass(frozen=True, slots=True)
class Papers:
    ids: list[str]
    vectors: Vectors
    texts: list[str]


@dataclass(frozen=True, slots=True)
class TopicSummary:
    topic_id: int | None  # None: papers that fit no topic
    papers: int
    keywords: list[str]


class TopicStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        conn.autocommit = True
        register_vector(conn)
        self._conn = conn

    def sample(self, model: str, since: date, size: int, seed: int) -> Papers:
        """Up to `size` papers in scope, the same ones for the same seed."""
        return self._papers(
            "SELECT p.arxiv_id, e.embedding, p.title || E'\\n\\n' || p.abstract "
            + _IN_SCOPE
            + " ORDER BY md5(e.arxiv_id || %(seed)s::text) LIMIT %(size)s",
            {"model": model, "since": since, "seed": seed, "size": size},
        )

    def unassigned(self, run: TopicRun, after: str, limit: int) -> Papers:
        """Papers in the run's scope that have no topic in it yet, ordered by id."""
        return self._papers(
            "SELECT p.arxiv_id, e.embedding, '' "
            + _IN_SCOPE
            + """ AND p.arxiv_id > %(after)s
                  AND NOT EXISTS (SELECT 1 FROM paper_topics pt
                                  WHERE pt.run_id = %(run)s AND pt.arxiv_id = p.arxiv_id)
                  ORDER BY p.arxiv_id LIMIT %(limit)s""",
            {
                "model": run.model,
                "since": run.published_since,
                "run": run.id,
                "after": after,
                "limit": limit,
            },
        )

    def save_fit(
        self,
        model: str,
        since: date,
        params: dict[str, object],
        papers: Papers,
        result: FitResult,
        keywords: dict[int, list[str]],
    ) -> int:
        """Store a fitted run in one transaction and return its id."""
        labels = result.labels
        with self._conn.transaction(), self._conn.cursor() as cur:
            row = cur.execute(
                """INSERT INTO topic_runs (model, published_since, params, sample_size)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (model, since, json.dumps(params), len(papers.ids)),
            ).fetchone()
            if row is None:
                raise RuntimeError("INSERT ... RETURNING returned no row")
            run_id = int(row[0])
            sizes = np.bincount(labels[labels >= 0], minlength=len(keywords))
            cur.executemany(
                "INSERT INTO topics (run_id, topic_id, size, keywords) VALUES (%s, %s, %s, %s)",
                [(run_id, t, int(sizes[t]), words) for t, words in sorted(keywords.items())],
            )
            self._copy_paper_topics(cur, run_id, papers.ids, labels, result.coords, fitted=True)
        return run_id

    def save_assigned(
        self, run_id: int, ids: list[str], labels: NDArray[np.int_], coords: NDArray[np.float32]
    ) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            self._copy_paper_topics(cur, run_id, ids, labels, coords, fitted=False)

    def run(self, run_id: int | None = None) -> TopicRun | None:
        """The given run, or the latest one."""
        row = self._conn.execute(
            """SELECT id, model, published_since FROM topic_runs
               WHERE %(id)s::bigint IS NULL OR id = %(id)s ORDER BY id DESC LIMIT 1""",
            {"id": run_id},
        ).fetchone()
        return TopicRun(*row) if row else None

    def reference(self, run_id: int) -> Reference:
        """The run's fitting sample, which new papers are compared against."""
        with self._conn.cursor(binary=True) as cur:
            rows = cur.execute(
                """SELECT e.embedding, pt.topic_id, pt.x, pt.y
                   FROM paper_topics pt
                   JOIN paper_embeddings e ON e.arxiv_id = pt.arxiv_id
                   WHERE pt.run_id = %s AND pt.fitted""",
                (run_id,),
            ).fetchall()
        return Reference(
            vectors=_stack([r[0] for r in rows]),
            labels=np.array([-1 if r[1] is None else r[1] for r in rows], dtype=np.int_),
            coords=np.array([(r[2], r[3]) for r in rows], dtype=np.float32).reshape(-1, 2),
        )

    def summary(self, run_id: int) -> list[TopicSummary]:
        """Every topic of the run with its paper count, largest first, then the rest."""
        rows = self._conn.execute(
            """SELECT pt.topic_id, count(*), coalesce(t.keywords, '{}')
               FROM paper_topics pt
               LEFT JOIN topics t ON t.run_id = pt.run_id AND t.topic_id = pt.topic_id
               WHERE pt.run_id = %s
               GROUP BY pt.topic_id, t.keywords
               ORDER BY pt.topic_id IS NULL, count(*) DESC""",
            (run_id,),
        ).fetchall()
        return [TopicSummary(*row) for row in rows]

    def _papers(self, query: LiteralString, params: dict[str, object]) -> Papers:
        # Binary results: parsing 1024 numbers per row from text would dominate the load.
        with self._conn.cursor(binary=True) as cur:
            rows = cur.execute(query, params).fetchall()
        return Papers(
            ids=[r[0] for r in rows],
            vectors=_stack([r[1] for r in rows]),
            texts=[r[2] for r in rows],
        )

    @staticmethod
    def _copy_paper_topics(
        cur: psycopg.Cursor,
        run_id: int,
        ids: list[str],
        labels: NDArray[np.int_],
        coords: NDArray[np.float32],
        fitted: bool,
    ) -> None:
        # COPY streams rows in bulk: far faster than INSERT for hundreds of thousands.
        with cur.copy(
            "COPY paper_topics (run_id, arxiv_id, topic_id, x, y, fitted) FROM STDIN"
        ) as copy:
            for arxiv_id, label, (x, y) in zip(ids, labels.tolist(), coords.tolist(), strict=True):
                copy.write_row((run_id, arxiv_id, None if label < 0 else label, x, y, fitted))


def _stack(halfvecs: list[HalfVector]) -> Vectors:
    if not halfvecs:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([v.to_numpy() for v in halfvecs]).astype(np.float32)
