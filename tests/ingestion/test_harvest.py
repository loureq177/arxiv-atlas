from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest

from arxiv_atlas.ingestion.harvest import Harvester, date_windows
from arxiv_atlas.ingestion.parser import OaiError, parse_list_records
from arxiv_atlas.ingestion.repository import HarvestRun

NOW = datetime(2026, 10, 26, 12, tzinfo=UTC)


@dataclass
class FakeRun:
    set_spec: str
    from_date: date
    until_date: date
    status: str = "running"
    token: str | None = None
    expires: datetime | None = None


class FakeRepo:
    def __init__(self) -> None:
        self.runs: dict[int, FakeRun] = {}
        self.saved: list[tuple[int, list[str]]] = []

    def _match(self, set_spec, from_date, until_date):
        return [
            (run_id, r)
            for run_id, r in self.runs.items()
            if (r.set_spec, r.from_date, r.until_date) == (set_spec, from_date, until_date)
        ]

    def is_window_done(self, set_spec, from_date, until_date):
        return any(r.status == "done" for _, r in self._match(set_spec, from_date, until_date))

    def find_unfinished_run(self, set_spec, from_date, until_date):
        for run_id, r in reversed(self._match(set_spec, from_date, until_date)):
            if r.status in ("running", "failed"):
                return HarvestRun(run_id, set_spec, from_date, until_date, r.token, r.expires)
        return None

    def start_run(self, set_spec, from_date, until_date):
        run_id = len(self.runs) + 1
        self.runs[run_id] = FakeRun(set_spec, from_date, until_date)
        return run_id

    def restart_run(self, run_id):
        run = self.runs[run_id]
        run.status, run.token, run.expires = "running", None, None

    def save_page(self, run_id, page, papers):
        run = self.runs[run_id]
        run.token, run.expires = page.resumption_token, page.token_expires_at
        self.saved.append((run_id, [p.arxiv_id for p in papers]))

    def finish_run(self, run_id):
        run = self.runs[run_id]
        run.status, run.token = "done", None

    def fail_run(self, run_id, error):
        self.runs[run_id].status = "failed"

    def last_done_until(self, set_spec):
        dates = [
            r.until_date
            for r in self.runs.values()
            if r.set_spec == set_spec and r.status == "done"
        ]
        return max(dates, default=None)


class FakeClient:
    """Serves the two-page fixture sequence: first page with a token, then the last page."""

    def __init__(self, fixture_bytes, fail_resume_with: Exception | None = None) -> None:
        self.first = parse_list_records(fixture_bytes("list_records_page.xml"))
        self.last = parse_list_records(fixture_bytes("last_page.xml"))
        self.fail_resume_with = fail_resume_with
        self.calls: list[tuple[object, ...]] = []

    def list_records(self, set_spec, from_date, until_date):
        self.calls.append(("list", set_spec, from_date, until_date))
        return self.first

    def resume(self, token):
        self.calls.append(("resume", token))
        if self.fail_resume_with is not None:
            exc, self.fail_resume_with = self.fail_resume_with, None
            raise exc
        return self.last


def make_harvester(client, repo, sets=("cs:cs:LG",)):
    return Harvester(client, repo, sets=sets, now=lambda: NOW)


def test_date_windows_are_inclusive_and_contiguous():
    assert date_windows(date(2026, 1, 1), date(2026, 1, 10), 4) == [
        (date(2026, 1, 1), date(2026, 1, 4)),
        (date(2026, 1, 5), date(2026, 1, 8)),
        (date(2026, 1, 9), date(2026, 1, 10)),
    ]


def test_harvest_window_follows_tokens_and_filters_categories(fixture_bytes):
    repo, client = FakeRepo(), FakeClient(fixture_bytes)

    make_harvester(client, repo).harvest_window("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))

    assert [c[0] for c in client.calls] == ["list", "resume"]
    # 2609.00002 is cs.CV only, so it is dropped.
    assert repo.saved == [(1, ["1906.08676"]), (1, ["2610.12345"])]
    assert repo.runs[1].status == "done"


def test_backfill_skips_done_windows(fixture_bytes):
    repo, client = FakeRepo(), FakeClient(fixture_bytes)
    harvester = make_harvester(client, repo)
    harvester.backfill(date(2026, 1, 1), date(2026, 1, 20), window_days=10)
    client.calls.clear()

    harvester.backfill(date(2026, 1, 1), date(2026, 1, 30), window_days=10)

    assert [c for c in client.calls if c[0] == "list"] == [
        ("list", "cs:cs:LG", date(2026, 1, 21), date(2026, 1, 30))
    ]


def test_crashed_run_resumes_from_saved_token(fixture_bytes):
    repo, client = FakeRepo(), FakeClient(fixture_bytes, fail_resume_with=RuntimeError("crash"))
    harvester = make_harvester(client, repo)
    window = ("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))

    with pytest.raises(RuntimeError):
        harvester.harvest_window(*window)
    assert repo.runs[1].status == "failed"
    client.calls.clear()

    harvester.harvest_window(*window)

    assert client.calls[0][0] == "resume"
    assert repo.runs[1].status == "done"


def test_expired_token_restarts_window(fixture_bytes):
    repo, client = FakeRepo(), FakeClient(fixture_bytes)
    run_id = repo.start_run("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))
    repo.runs[run_id].token = "old"
    repo.runs[run_id].expires = datetime(2026, 10, 1, tzinfo=UTC)

    make_harvester(client, repo).harvest_window("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))

    assert client.calls[0][0] == "list"


def test_bad_resumption_token_restarts_window(fixture_bytes):
    repo = FakeRepo()
    client = FakeClient(fixture_bytes, fail_resume_with=OaiError("badResumptionToken", "x"))

    make_harvester(client, repo).harvest_window("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))

    assert [c[0] for c in client.calls] == ["list", "resume", "list", "resume"]
    assert repo.runs[1].status == "done"


def test_update_continues_from_last_done_day(fixture_bytes):
    repo, client = FakeRepo(), FakeClient(fixture_bytes)
    run_id = repo.start_run("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 20))
    repo.finish_run(run_id)

    make_harvester(client, repo, sets=("cs:cs:LG", "cs:cs:CL")).update()

    lists = [c for c in client.calls if c[0] == "list"]
    # cs:cs:CL was never backfilled, so update leaves it alone.
    assert lists == [("list", "cs:cs:LG", date(2026, 10, 20), date(2026, 10, 26))]
