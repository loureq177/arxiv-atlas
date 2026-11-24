from datetime import date

import httpx
import pytest

from arxiv_atlas.ingestion.oai_client import OaiClient, OaiUnavailableError


class FakeTime:
    """Deterministic clock: sleeping just advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def make_client(handler, fake_time, **kwargs) -> OaiClient:
    return OaiClient(
        transport=httpx.MockTransport(handler),
        sleep=fake_time.sleep,
        clock=fake_time.clock,
        **kwargs,
    )


def test_list_records_sends_selective_harvest_params(fixture_bytes):
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, content=fixture_bytes("last_page.xml"))

    client = make_client(handler, FakeTime())
    page = client.list_records("cs:cs:LG", date(2026, 1, 1), date(2026, 1, 31))

    params = seen[0].params
    assert params["verb"] == "ListRecords"
    assert params["metadataPrefix"] == "arXivRaw"
    assert params["set"] == "cs:cs:LG"
    assert (params["from"], params["until"]) == ("2026-01-01", "2026-01-31")
    assert len(page.papers) == 1


def test_resumption_token_is_sent_verbatim_after_decoding(fixture_bytes):
    token = "verb%3DListRecords%26skip%3D48"
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, content=fixture_bytes("last_page.xml"))

    make_client(handler, FakeTime()).resume(token)

    # The server URL-decodes the query once and must get the token back unchanged.
    assert seen[0].params["resumptionToken"] == token
    assert "verb" in seen[0].params
    assert "metadataPrefix" not in seen[0].params


def test_enforces_minimum_interval_between_requests(fixture_bytes):
    fake_time = FakeTime()

    def handler(request):
        fake_time.now += 1.0  # each request takes 1s
        return httpx.Response(200, content=fixture_bytes("last_page.xml"))

    client = make_client(handler, fake_time)
    client.resume("t1")
    client.resume("t2")

    assert fake_time.sleeps == [pytest.approx(2.0)]


def test_retries_503_honouring_retry_after(fixture_bytes):
    responses = iter(
        [
            httpx.Response(503, headers={"Retry-After": "17"}),
            httpx.Response(200, content=fixture_bytes("last_page.xml")),
        ]
    )
    fake_time = FakeTime()
    client = make_client(lambda request: next(responses), fake_time)

    page = client.resume("t")

    assert len(page.papers) == 1
    assert 17.0 in fake_time.sleeps


def test_retries_transport_errors_with_backoff(fixture_bytes):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, content=fixture_bytes("last_page.xml"))

    fake_time = FakeTime()
    make_client(handler, fake_time).resume("t")

    assert calls["n"] == 3
    assert fake_time.sleeps[:2] == [5.0, 10.0]


def test_gives_up_after_max_retries():
    fake_time = FakeTime()
    client = make_client(lambda request: httpx.Response(503), fake_time, max_retries=2)

    with pytest.raises(OaiUnavailableError):
        client.resume("t")


def test_client_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(handler, FakeTime()).resume("t")
    assert calls["n"] == 1


def test_rejects_interval_below_arxiv_policy():
    with pytest.raises(ValueError, match="between requests"):
        OaiClient(min_interval=1.0)
