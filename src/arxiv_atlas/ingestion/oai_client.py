"""HTTP client for arXiv's OAI-PMH endpoint: rate limiting and retries.

arXiv asks for at most one request every 3 seconds and answers overload with
`503 Retry-After`. Both are handled here so the rest of the code can treat a page
fetch as a plain function call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime

import httpx

from arxiv_atlas.ingestion.parser import OaiPage, parse_list_records

log = logging.getLogger(__name__)

BASE_URL = "https://oaipmh.arxiv.org/oai"
METADATA_PREFIX = "arXivRaw"
MIN_INTERVAL_S = 3.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_S = 600.0


class OaiUnavailableError(Exception):
    """The endpoint kept failing after all retries."""


class OaiClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        contact: str | None = None,
        min_interval: float = MIN_INTERVAL_S,
        max_retries: int = 8,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_interval < MIN_INTERVAL_S:
            raise ValueError(f"arXiv requires >= {MIN_INTERVAL_S}s between requests")
        user_agent = "arxiv-atlas/0.1" + (f" (+{contact})" if contact else "")
        self._base_url = base_url
        self._http = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def __enter__(self) -> OaiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def list_records(self, set_spec: str, from_date: date, until_date: date) -> OaiPage:
        """First page of a selective harvest; `from` and `until` are inclusive."""
        return parse_list_records(
            self._get(
                {
                    "verb": "ListRecords",
                    "metadataPrefix": METADATA_PREFIX,
                    "set": set_spec,
                    "from": from_date.isoformat(),
                    "until": until_date.isoformat(),
                }
            )
        )

    def resume(self, resumption_token: str) -> OaiPage:
        # The token is opaque and already contains %-escapes; httpx escapes it once
        # more, which is what the server expects.
        return parse_list_records(
            self._get({"verb": "ListRecords", "resumptionToken": resumption_token})
        )

    def _get(self, params: dict[str, str]) -> bytes:
        attempt = 0
        while True:
            self._wait_for_slot()
            try:
                response = self._http.get(self._base_url, params=params)
            except httpx.TransportError as exc:
                problem, delay = repr(exc), self._backoff(attempt)
            else:
                if response.status_code == 200:
                    return response.content
                if response.status_code not in RETRYABLE_STATUS:
                    response.raise_for_status()
                problem = f"HTTP {response.status_code}"
                delay = _retry_after_seconds(response) or self._backoff(attempt)

            if attempt >= self._max_retries:
                raise OaiUnavailableError(f"giving up after {attempt + 1} attempts: {problem}")
            log.warning("%s, retrying in %.0fs (attempt %d)", problem, delay, attempt + 1)
            self._sleep(delay)
            attempt += 1

    def _wait_for_slot(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._last_request_at + self._min_interval - now
            if wait > 0:
                self._sleep(wait)
                now += wait
        self._last_request_at = now

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(MAX_BACKOFF_S, 5.0 * 2.0**attempt)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    if value.strip().isdigit():
        seconds = float(value)
    else:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except TypeError, ValueError:
            return None
    return min(MAX_BACKOFF_S, max(seconds, 1.0))
