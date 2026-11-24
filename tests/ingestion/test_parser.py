from datetime import UTC, date, datetime

import pytest

from arxiv_atlas.ingestion.parser import OaiError, parse_list_records


def test_parses_papers_deleted_records_and_token(fixture_bytes):
    page = parse_list_records(fixture_bytes("list_records_page.xml"))

    assert [p.arxiv_id for p in page.papers] == ["1906.08676", "2609.00002"]
    assert page.deleted_ids == ["2401.00001"]
    assert page.records_seen == 3
    assert page.resumption_token is not None
    assert page.resumption_token.startswith("verb%3DListRecords")
    assert page.token_expires_at == datetime(2026, 10, 27, tzinfo=UTC)


def test_uses_v1_date_not_latest_version(fixture_bytes):
    paper = parse_list_records(fixture_bytes("list_records_page.xml")).papers[0]

    assert paper.published_at == datetime(2019, 6, 20, 14, 54, 14, tzinfo=UTC)
    assert paper.updated_at == datetime(2021, 1, 27, 1, 55, 50, tzinfo=UTC)
    assert paper.num_versions == 2
    assert paper.oai_datestamp == date(2026, 10, 20)


def test_normalises_text_and_categories(fixture_bytes):
    paper = parse_list_records(fixture_bytes("list_records_page.xml")).papers[0]

    assert paper.title == (
        "Homogeneous Vector Capsules Enable Adaptive Gradient Descent in "
        "Convolutional Neural Networks"
    )
    assert "\n" not in paper.abstract
    assert "model's achievable" in paper.abstract
    assert paper.categories == ("cs.CV", "cs.LG")
    assert paper.primary_category == "cs.CV"
    assert paper.doi == "10.1109/ACCESS.2021.3066842"


def test_optional_fields_are_none_when_missing(fixture_bytes):
    paper = parse_list_records(fixture_bytes("last_page.xml")).papers[0]

    assert paper.comments is None
    assert paper.doi is None
    assert paper.license is None


def test_empty_resumption_token_marks_last_page(fixture_bytes):
    page = parse_list_records(fixture_bytes("last_page.xml"))

    assert page.resumption_token is None
    assert page.token_expires_at is None


def test_no_records_match_is_an_empty_page(fixture_bytes):
    page = parse_list_records(fixture_bytes("error_no_records.xml"))

    assert page.papers == []
    assert page.deleted_ids == []
    assert page.resumption_token is None


def test_other_protocol_errors_raise(fixture_bytes):
    with pytest.raises(OaiError) as excinfo:
        parse_list_records(fixture_bytes("error_bad_argument.xml"))

    assert excinfo.value.code == "badArgument"


def test_malformed_record_is_skipped_not_fatal(fixture_bytes):
    xml = fixture_bytes("last_page.xml").replace(b"<categories>cs.CL cs.AI</categories>", b"")

    assert parse_list_records(xml).papers == []
