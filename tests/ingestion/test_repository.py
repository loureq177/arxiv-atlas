"""Integration tests against a real PostgreSQL; see tests/conftest.py."""

import dataclasses
from datetime import date

import pytest

from arxiv_atlas.ingestion.parser import parse_list_records
from arxiv_atlas.ingestion.repository import Repository


@pytest.fixture
def repo(db):
    return Repository(db)


def fetch_paper(repo, arxiv_id):
    return repo._conn.execute(
        "SELECT title, primary_category, categories, published_at, is_deleted"
        " FROM papers WHERE arxiv_id = %s",
        (arxiv_id,),
    ).fetchone()


def test_save_page_upserts_and_checkpoints(repo, fixture_bytes):
    page = parse_list_records(fixture_bytes("list_records_page.xml"))
    run_id = repo.start_run("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))

    repo.save_page(run_id, page, page.papers[:1])
    repo.save_page(run_id, page, page.papers[:1])  # replaying a page is harmless

    _, primary, categories, published_at, deleted = fetch_paper(repo, "1906.08676")
    assert primary == "cs.CV"
    assert categories == ["cs.CV", "cs.LG"]
    assert published_at.year == 2019
    assert not deleted
    run = repo.find_unfinished_run("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))
    assert run.resumption_token == page.resumption_token
    count = repo._conn.execute("SELECT count(*) FROM papers").fetchone()[0]
    assert count == 1


def test_older_datestamp_does_not_overwrite_newer(repo, fixture_bytes):
    page = parse_list_records(fixture_bytes("list_records_page.xml"))
    paper = page.papers[0]
    stale = dataclasses.replace(paper, title="Stale title", oai_datestamp=date(2020, 1, 1))
    run_id = repo.start_run("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))

    repo.save_page(run_id, page, [paper])
    repo.save_page(run_id, page, [stale])

    assert fetch_paper(repo, paper.arxiv_id)[0] == paper.title


def test_deleted_records_are_flagged(repo, fixture_bytes):
    page = parse_list_records(fixture_bytes("list_records_page.xml"))
    deleted_id = page.deleted_ids[0]
    alive = dataclasses.replace(page.papers[0], arxiv_id=deleted_id)
    run_id = repo.start_run("cs:cs:LG", date(2026, 10, 1), date(2026, 10, 25))
    no_deletions = dataclasses.replace(page, deleted_ids=[])

    repo.save_page(run_id, no_deletions, [alive])
    repo.save_page(run_id, page, [])

    assert fetch_paper(repo, deleted_id)[4] is True


def test_run_lifecycle(repo):
    window = ("stat:stat:ML", date(2026, 1, 1), date(2026, 1, 31))
    run_id = repo.start_run(*window)
    assert not repo.is_window_done(*window)

    repo.finish_run(run_id)

    assert repo.is_window_done(*window)
    assert repo.find_unfinished_run(*window) is None
    assert repo.last_done_until("stat:stat:ML") == date(2026, 1, 31)
