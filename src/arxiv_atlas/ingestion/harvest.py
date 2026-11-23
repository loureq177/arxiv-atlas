"""Harvest orchestration: split work into datestamp windows, resume after crashes.

OAI-PMH filters by *datestamp* (last metadata change), not by submission date. Every
record has exactly one current datestamp, so covering all windows from arXiv's
earliest datestamp to today yields every record; a record modified mid-backfill moves
to a recent window and is picked up there or by the next `update`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

from arxiv_atlas.ingestion.oai_client import OaiClient
from arxiv_atlas.ingestion.parser import OaiError, OaiPage, Paper
from arxiv_atlas.ingestion.repository import HarvestRun

log = logging.getLogger(__name__)

# arXiv exposes per-subject sets, so we never download all of cs.
TARGET_SETS: dict[str, str] = {
    "cs:cs:LG": "cs.LG",
    "cs:cs:CL": "cs.CL",
    "cs:cs:AI": "cs.AI",
    "stat:stat:ML": "stat.ML",
}
TARGET_CATEGORIES = frozenset(TARGET_SETS.values())
EARLIEST_DATESTAMP = date(2005, 9, 16)  # from the endpoint's Identify response


class HarvestRepository(Protocol):
    def is_window_done(self, set_spec: str, from_date: date, until_date: date) -> bool: ...
    def find_unfinished_run(
        self, set_spec: str, from_date: date, until_date: date
    ) -> HarvestRun | None: ...
    def start_run(self, set_spec: str, from_date: date, until_date: date) -> int: ...
    def restart_run(self, run_id: int) -> None: ...
    def save_page(self, run_id: int, page: OaiPage, papers: list[Paper]) -> None: ...
    def finish_run(self, run_id: int) -> None: ...
    def fail_run(self, run_id: int, error: str) -> None: ...
    def last_done_until(self, set_spec: str) -> date | None: ...


def date_windows(start: date, end: date, days: int) -> list[tuple[date, date]]:
    """Consecutive, non-overlapping, inclusive windows covering [start, end]."""
    if days < 1:
        raise ValueError("window must be at least one day")
    windows = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=days - 1), end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def is_relevant(paper: Paper, categories: Iterable[str] = TARGET_CATEGORIES) -> bool:
    return not set(paper.categories).isdisjoint(categories)


class Harvester:
    def __init__(
        self,
        client: OaiClient,
        repo: HarvestRepository,
        sets: Iterable[str] = TARGET_SETS,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._repo = repo
        self._sets = list(sets)
        self._now = now

    def today(self) -> date:
        return self._now().date()

    def backfill(self, start: date, end: date, window_days: int) -> None:
        windows = date_windows(start, end, window_days)
        for set_spec in self._sets:
            for from_date, until_date in windows:
                if self._repo.is_window_done(set_spec, from_date, until_date):
                    continue
                self.harvest_window(set_spec, from_date, until_date)

    def update(self) -> None:
        """Fetch everything changed since the last completed window, per set."""
        today = self.today()
        for set_spec in self._sets:
            last = self._repo.last_done_until(set_spec)
            if last is None:
                log.warning("%s has never been harvested; run `backfill` first", set_spec)
                continue
            # Re-fetch the last day too: records stamped with it may have arrived
            # after that run finished. Upserts make the overlap harmless.
            self.harvest_window(set_spec, last, today)

    def harvest_window(self, set_spec: str, from_date: date, until_date: date) -> None:
        run = self._repo.find_unfinished_run(set_spec, from_date, until_date)
        if run is None:
            run_id = self._repo.start_run(set_spec, from_date, until_date)
            token = None
        else:
            run_id = run.id
            token = run.resumption_token if self._token_usable(run) else None
            if token is None:
                self._repo.restart_run(run_id)

        log.info(
            "%s %s..%s: %s",
            set_spec, from_date, until_date, "resuming" if token else "starting",
        )
        try:
            page = self._fetch(set_spec, from_date, until_date, token, run_id)
            while True:
                relevant = [p for p in page.papers if is_relevant(p)]
                self._repo.save_page(run_id, page, relevant)
                log.info(
                    "%s %s..%s: page with %d records, %d kept",
                    set_spec, from_date, until_date, page.records_seen, len(relevant),
                )
                if page.resumption_token is None:
                    break
                page = self._fetch(set_spec, from_date, until_date, page.resumption_token, run_id)
        except BaseException as exc:
            self._repo.fail_run(run_id, repr(exc))
            raise
        self._repo.finish_run(run_id)

    def _fetch(
        self, set_spec: str, from_date: date, until_date: date, token: str | None, run_id: int
    ) -> OaiPage:
        if token is None:
            return self._client.list_records(set_spec, from_date, until_date)
        try:
            return self._client.resume(token)
        except OaiError as exc:
            if exc.code != "badResumptionToken":
                raise
            # Tokens expire; the window restarts from scratch and upserts dedupe it.
            log.warning("resumption token rejected, restarting window: %s", exc)
            self._repo.restart_run(run_id)
            return self._client.list_records(set_spec, from_date, until_date)

    def _token_usable(self, run: HarvestRun) -> bool:
        if run.resumption_token is None:
            return False
        return run.token_expires_at is None or run.token_expires_at > self._now()
