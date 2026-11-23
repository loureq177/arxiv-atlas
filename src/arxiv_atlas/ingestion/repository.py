"""PostgreSQL persistence for papers and harvest checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from importlib.resources import files

import psycopg

from arxiv_atlas.ingestion.parser import OaiPage, Paper

# Re-harvesting an old window must not overwrite fresher metadata, hence the
# datestamp guard in the WHERE clause.
UPSERT_PAPER = """
INSERT INTO papers (
    arxiv_id, title, abstract, authors, submitter, primary_category, categories,
    published_at, updated_at, num_versions, comments, journal_ref, doi, license,
    oai_datestamp
) VALUES (
    %(arxiv_id)s, %(title)s, %(abstract)s, %(authors)s, %(submitter)s,
    %(primary_category)s, %(categories)s, %(published_at)s, %(updated_at)s,
    %(num_versions)s, %(comments)s, %(journal_ref)s, %(doi)s, %(license)s,
    %(oai_datestamp)s
)
ON CONFLICT (arxiv_id) DO UPDATE SET
    title            = EXCLUDED.title,
    abstract         = EXCLUDED.abstract,
    authors          = EXCLUDED.authors,
    submitter        = EXCLUDED.submitter,
    primary_category = EXCLUDED.primary_category,
    categories       = EXCLUDED.categories,
    published_at     = EXCLUDED.published_at,
    updated_at       = EXCLUDED.updated_at,
    num_versions     = EXCLUDED.num_versions,
    comments         = EXCLUDED.comments,
    journal_ref      = EXCLUDED.journal_ref,
    doi              = EXCLUDED.doi,
    license          = EXCLUDED.license,
    oai_datestamp    = EXCLUDED.oai_datestamp,
    is_deleted       = false,
    last_seen_at     = now()
WHERE papers.oai_datestamp <= EXCLUDED.oai_datestamp
"""


@dataclass(frozen=True, slots=True)
class HarvestRun:
    id: int
    set_spec: str
    from_date: date
    until_date: date
    resumption_token: str | None
    token_expires_at: datetime | None


class Repository:
    def __init__(self, conn: psycopg.Connection) -> None:
        # Autocommit everywhere except inside explicit `transaction()` blocks.
        conn.autocommit = True
        self._conn = conn

    @classmethod
    def connect(cls, dsn: str) -> Repository:
        return cls(psycopg.connect(dsn))

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        sql = files("arxiv_atlas.db").joinpath("migrations", "0001_ingestion.sql").read_text()
        self._conn.execute(sql)

    def is_window_done(self, set_spec: str, from_date: date, until_date: date) -> bool:
        row = self._conn.execute(
            """SELECT 1 FROM harvest_runs
               WHERE set_spec = %s AND from_date = %s AND until_date = %s AND status = 'done'
               LIMIT 1""",
            (set_spec, from_date, until_date),
        ).fetchone()
        return row is not None

    def find_unfinished_run(
        self, set_spec: str, from_date: date, until_date: date
    ) -> HarvestRun | None:
        row = self._conn.execute(
            """SELECT id, set_spec, from_date, until_date, resumption_token, token_expires_at
               FROM harvest_runs
               WHERE set_spec = %s AND from_date = %s AND until_date = %s
                 AND status IN ('running', 'failed')
               ORDER BY id DESC LIMIT 1""",
            (set_spec, from_date, until_date),
        ).fetchone()
        return HarvestRun(*row) if row else None

    def start_run(self, set_spec: str, from_date: date, until_date: date) -> int:
        row = self._conn.execute(
            """INSERT INTO harvest_runs (set_spec, from_date, until_date, status)
               VALUES (%s, %s, %s, 'running') RETURNING id""",
            (set_spec, from_date, until_date),
        ).fetchone()
        return row[0]

    def restart_run(self, run_id: int) -> None:
        self._conn.execute(
            """UPDATE harvest_runs
               SET status = 'running', resumption_token = NULL, token_expires_at = NULL,
                   error = NULL, pages_fetched = 0, records_seen = 0, records_upserted = 0
               WHERE id = %s""",
            (run_id,),
        )

    def save_page(self, run_id: int, page: OaiPage, papers: list[Paper]) -> None:
        """Store a page and advance the checkpoint atomically.

        If the process dies mid-page, neither the rows nor the new token are committed,
        so the next run re-fetches exactly this page.
        """
        with self._conn.transaction(), self._conn.cursor() as cur:
            if papers:
                cur.executemany(UPSERT_PAPER, [_paper_params(p) for p in papers])
            if page.deleted_ids:
                cur.execute(
                    """UPDATE papers SET is_deleted = true, last_seen_at = now()
                       WHERE arxiv_id = ANY(%s)""",
                    (page.deleted_ids,),
                )
            cur.execute(
                """UPDATE harvest_runs
                   SET resumption_token = %s, token_expires_at = %s,
                       pages_fetched = pages_fetched + 1,
                       records_seen = records_seen + %s,
                       records_upserted = records_upserted + %s
                   WHERE id = %s""",
                (
                    page.resumption_token,
                    page.token_expires_at,
                    page.records_seen,
                    len(papers),
                    run_id,
                ),
            )

    def finish_run(self, run_id: int) -> None:
        self._conn.execute(
            """UPDATE harvest_runs
               SET status = 'done', resumption_token = NULL, token_expires_at = NULL,
                   finished_at = now()
               WHERE id = %s""",
            (run_id,),
        )

    def fail_run(self, run_id: int, error: str) -> None:
        self._conn.execute(
            "UPDATE harvest_runs SET status = 'failed', error = %s WHERE id = %s",
            (error, run_id),
        )

    def last_done_until(self, set_spec: str) -> date | None:
        row = self._conn.execute(
            "SELECT max(until_date) FROM harvest_runs WHERE set_spec = %s AND status = 'done'",
            (set_spec,),
        ).fetchone()
        return row[0]

    def status(self) -> dict[str, object]:
        papers = self._conn.execute(
            """SELECT count(*), count(*) FILTER (WHERE is_deleted),
                      min(published_at), max(published_at)
               FROM papers"""
        ).fetchone()
        by_category = self._conn.execute(
            """SELECT primary_category, count(*) FROM papers
               WHERE NOT is_deleted GROUP BY 1 ORDER BY 2 DESC LIMIT 15"""
        ).fetchall()
        runs = self._conn.execute(
            """SELECT set_spec, status, count(*), max(until_date)
               FROM harvest_runs GROUP BY 1, 2 ORDER BY 1, 2"""
        ).fetchall()
        return {"papers": papers, "by_primary_category": by_category, "runs": runs}


def _paper_params(paper: Paper) -> dict[str, object]:
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "submitter": paper.submitter,
        "primary_category": paper.primary_category,
        # psycopg adapts lists, not tuples, to Postgres arrays.
        "categories": list(paper.categories),
        "published_at": paper.published_at,
        "updated_at": paper.updated_at,
        "num_versions": paper.num_versions,
        "comments": paper.comments,
        "journal_ref": paper.journal_ref,
        "doi": paper.doi,
        "license": paper.license,
        "oai_datestamp": paper.oai_datestamp,
    }
