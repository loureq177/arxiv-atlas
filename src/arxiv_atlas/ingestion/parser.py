"""Parse OAI-PMH ListRecords responses in the arXivRaw metadata format.

arXivRaw is used instead of the plain `arXiv` format because only arXivRaw carries
the per-version history. The `<created>` field of the `arXiv` format is NOT reliably
the v1 date (e.g. 1906.08676 reports 2021-01-27, the date of v2), and trend analysis
needs the real first-submission date.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from email.utils import parsedate_to_datetime

log = logging.getLogger(__name__)

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}
OAI_ID_PREFIX = "oai:arXiv.org:"


@dataclass(frozen=True, slots=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    submitter: str | None
    categories: tuple[str, ...]
    published_at: datetime
    updated_at: datetime
    num_versions: int
    comments: str | None
    journal_ref: str | None
    doi: str | None
    license: str | None
    oai_datestamp: date

    @property
    def primary_category(self) -> str:
        # arXiv lists the primary category first.
        return self.categories[0]


@dataclass(frozen=True, slots=True)
class OaiPage:
    papers: list[Paper]
    deleted_ids: list[str]
    resumption_token: str | None
    token_expires_at: datetime | None

    @property
    def records_seen(self) -> int:
        return len(self.papers) + len(self.deleted_ids)


EMPTY_PAGE = OaiPage(papers=[], deleted_ids=[], resumption_token=None, token_expires_at=None)


class OaiError(Exception):
    """An OAI-PMH protocol error (the `<error code=...>` element)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def parse_list_records(xml: bytes) -> OaiPage:
    root = ET.fromstring(xml)

    error = root.find("oai:error", NS)
    if error is not None:
        code = error.get("code", "unknown")
        # An empty window is a normal outcome, not a failure.
        if code == "noRecordsMatch":
            return EMPTY_PAGE
        raise OaiError(code, _clean(error.text) or "")

    list_records = root.find("oai:ListRecords", NS)
    if list_records is None:
        raise OaiError("malformedResponse", "neither <ListRecords> nor <error> in response")

    papers: list[Paper] = []
    deleted_ids: list[str] = []
    for record in list_records.iterfind("oai:record", NS):
        header = record.find("oai:header", NS)
        arxiv_id = _strip_prefix(header.findtext("oai:identifier", default="", namespaces=NS))
        if header.get("status") == "deleted":
            deleted_ids.append(arxiv_id)
            continue
        datestamp = date.fromisoformat(header.findtext("oai:datestamp", namespaces=NS).strip())
        metadata = record.find("oai:metadata/raw:arXivRaw", NS)
        try:
            papers.append(_parse_paper(metadata, datestamp))
        except (AttributeError, ValueError, TypeError) as exc:
            # One broken record should not abort a multi-hour harvest.
            log.warning("skipping malformed record %s: %r", arxiv_id, exc)

    token, expires_at = _parse_token(list_records.find("oai:resumptionToken", NS))
    return OaiPage(papers, deleted_ids, token, expires_at)


def _parse_paper(meta: ET.Element, datestamp: date) -> Paper:
    version_dates = [
        parsedate_to_datetime(v.findtext("raw:date", namespaces=NS))
        for v in meta.iterfind("raw:version", NS)
    ]
    if not version_dates:
        raise ValueError("record has no <version> entries")

    categories = tuple(_field(meta, "categories").split())
    if not categories:
        raise ValueError("record has no categories")

    return Paper(
        arxiv_id=_field(meta, "id"),
        title=_field(meta, "title"),
        abstract=_field(meta, "abstract"),
        authors=_field(meta, "authors"),
        submitter=_optional(meta, "submitter"),
        categories=categories,
        published_at=min(version_dates),
        updated_at=max(version_dates),
        num_versions=len(version_dates),
        comments=_optional(meta, "comments"),
        journal_ref=_optional(meta, "journal-ref"),
        doi=_optional(meta, "doi"),
        license=_optional(meta, "license"),
        oai_datestamp=datestamp,
    )


def _parse_token(element: ET.Element | None) -> tuple[str | None, datetime | None]:
    # OAI-PMH marks the last page with an empty (or missing) <resumptionToken/>.
    if element is None or not (element.text or "").strip():
        return None, None
    expires = element.get("expirationDate")
    return element.text.strip(), datetime.fromisoformat(expires) if expires else None


def _field(meta: ET.Element, tag: str) -> str:
    value = _clean(meta.findtext(f"raw:{tag}", namespaces=NS))
    if not value:
        raise ValueError(f"missing <{tag}>")
    return value


def _optional(meta: ET.Element, tag: str) -> str | None:
    return _clean(meta.findtext(f"raw:{tag}", namespaces=NS)) or None


def _clean(text: str | None) -> str:
    # arXiv hard-wraps titles and abstracts; collapse all whitespace runs.
    return " ".join(text.split()) if text else ""


def _strip_prefix(identifier: str) -> str:
    return identifier.strip().removeprefix(OAI_ID_PREFIX)
