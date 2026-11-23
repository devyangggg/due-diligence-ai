"""Offline SEC and retrieval checks; fixture content is synthetic, not an SEC filing."""

from pathlib import Path

import httpx
import numpy as np
import pytest

import data
from retrieval import LocalEmbedder, Passage, make_passages
from scripts.hybrid_experiment import HybridRetriever

FIXTURES = Path(__file__).parent / "fixtures"
ACCESSION = "0000012345-26-000001"


def fixture(name):
    return (FIXTURES / name).read_bytes()


def sec_transport(html="filing.html", submissions="submissions.json", on_request=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if on_request:
            on_request(request)
        path = request.url.path
        if path.endswith("company_tickers.json"):
            return httpx.Response(200, content=fixture("company_tickers.json"))
        if path.endswith("CIK0000012345.json"):
            return httpx.Response(200, content=fixture(submissions))
        if path.endswith("index.json"):
            return httpx.Response(200, content=fixture("index.json"))
        if path.endswith("acme-2026x10k.htm"):
            return httpx.Response(200, content=fixture(html))
        raise AssertionError(f"Unexpected external URL: {request.url}")
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def no_sec_delay(monkeypatch):
    monkeypatch.setattr(data, "_SEC_INTERVAL", 0)
    monkeypatch.setattr(data, "_SEC_NEXT_REQUEST", 0)
    monkeypatch.setattr(data, "_CIK_MAP", {})
    monkeypatch.setattr(data, "_CIK_MAP_EXPIRES", 0)


@pytest.mark.asyncio
async def test_resolves_latest_filing_and_excludes_toc_and_neighbor_items():
    requested = []
    async with httpx.AsyncClient(transport=sec_transport(on_request=requested.append)) as http:
        sec = data.SECClient(http, "DueDiligenceAI tests@example.com")
        ref = await sec.resolve_filing("acme")
        filing = await sec.fetch_filing(ref)
    assert ref.accession == ACCESSION
    assert ref.cik == "0000012345"
    assert ref.url.endswith("/acme-2026x10k.htm")
    assert list(filing.sections) == ["Item 1", "Item 1A", "Item 7", "Item 8"]
    assert "Acme makes industrial sensors" in filing.sections["Item 1"]
    assert "one supplier" in filing.sections["Item 1A"]
    assert "Revenue rose 8 percent" in filing.sections["Item 7"]
    assert "Revenue was 108 million dollars" in filing.sections["Item 8"]
    assert "Two factories" not in filing.sections["Item 1A"]
    assert "Not part of Item 8" not in filing.sections["Item 8"]
    assert len(requested) == 4
    assert all("tests@example.com" in request.headers["User-Agent"] for request in requested)


@pytest.mark.asyncio
async def test_missing_heading_is_an_explicit_error():
    async with httpx.AsyncClient(transport=sec_transport(html="missing_item.html")) as http:
        sec = data.SECClient(http, "DueDiligenceAI tests@example.com")
        ref = await sec.resolve_filing("ACME")
        with pytest.raises(ValueError, match="Item 8"):
            await sec.fetch_filing(ref)


@pytest.mark.asyncio
async def test_running_page_headers_do_not_truncate_items():
    async with httpx.AsyncClient(transport=sec_transport(html="paged_filing.html")) as http:
        sec = data.SECClient(http, "DueDiligenceAI tests@example.com")
        filing = await sec.fetch_filing(await sec.resolve_filing("ACME"))
    for item, first, second in (
        ("Item 1", "First-page business", "Second-page business"),
        ("Item 1A", "First-page risk", "Second-page risk"),
        ("Item 7", "First-page discussion", "Second-page discussion"),
        ("Item 8", "First-page statement", "Second-page statement"),
    ):
        assert first in filing.sections[item]
        assert second in filing.sections[item]
    assert "Not Item 7" not in filing.sections["Item 7"]
    assert "Not Item 8" not in filing.sections["Item 8"]


@pytest.mark.asyncio
async def test_transient_sec_429_retries_but_permanent_4xx_does_not(monkeypatch):
    monkeypatch.setattr(data, "_retry_delay", lambda header, attempt: 0)
    statuses = [429, 200]
    def handler(request):
        return httpx.Response(statuses.pop(0), json={"ok": True})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert (await data.SECClient(http, "App tests@example.com")._json("https://www.sec.gov/test"))["ok"]
    assert not statuses
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404))) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await data.SECClient(http, "App tests@example.com")._json("https://www.sec.gov/test")


@pytest.mark.asyncio
async def test_sec_timeout_is_bounded_and_missing_user_agent_rejected(monkeypatch):
    monkeypatch.setattr(data, "_MAX_RETRIES", 0)

    def timeout(_request):
        raise httpx.ReadTimeout("offline timeout")

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as http:
        with pytest.raises(httpx.ReadTimeout):
            await data.SECClient(http, "App tests@example.com")._json("https://www.sec.gov/test")
        with pytest.raises(ValueError, match="User-Agent"):
            await data.SECClient(http, "")._json("https://www.sec.gov/test")


def test_passages_preserve_source_and_chunk_bound():
    source = "\n\n".join(f"Sensor paragraph {i}: " + " ".join(["industrial"] * 60) for i in range(16))
    passages = make_passages(ACCESSION, {"Item 1": source})
    assert len(passages) > 1
    assert all(p.text in source for p in passages)
    assert all(len(p.text.split()) <= 350 for p in passages)
    assert [p.index for p in passages] == list(range(len(passages)))
    assert passages == make_passages(ACCESSION, {"Item 1": source})


@pytest.mark.asyncio
async def test_cosine_ranking_is_restricted_to_requested_item():
    class FakeEmbedder(LocalEmbedder):
        async def embed(self, texts):
            return [np.array([1., 0.], dtype="<f4").tobytes() for _ in texts]

    source = [
        Passage("a", ACCESSION, "Item 1", 0, "business", np.array([0., 1.], dtype="<f4").tobytes()),
        Passage("b", ACCESSION, "Item 1", 1, "relevant", np.array([1., 0.], dtype="<f4").tobytes()),
        Passage("c", ACCESSION, "Item 1A", 0, "different item", np.array([1., 0.], dtype="<f4").tobytes()),
    ]
    ranked = await FakeEmbedder().search("relevant business", source, "Item 1", k=1)
    assert [passage.id for passage in ranked] == ["b"]


@pytest.mark.asyncio
async def test_hybrid_ranking_recovers_exact_financial_term_without_crossing_items():
    class SemanticOnlyEmbedder(LocalEmbedder):
        async def embed(self, texts):
            return [np.array([1., 0.], dtype="<f4").tobytes() for _ in texts]

    def vector(x, y):
        return np.array([x, y], dtype="<f4").tobytes()

    passages = [
        Passage("distractor", ACCESSION, "Item 8", 0, "Revenue forecast and product outlook.",
                vector(1., 0.)),
        Passage("net-income", ACCESSION, "Item 8", 1, "Consolidated net income was $12 million.",
                vector(.8, .2)),
        Passage("note", ACCESSION, "Item 8", 2, "Notes to the financial statements.",
                vector(.7, .3)),
        Passage("other-item", ACCESSION, "Item 7", 0, "Consolidated net income was $99 million.",
                vector(1., 0.)),
    ]
    embedder = SemanticOnlyEmbedder()
    assert (await embedder.search("What was net income?", passages, "Item 8", k=1))[0].id == "distractor"
    ranking = HybridRetriever(embedder, passages)
    assert [p.id for p in await ranking.search("What was net income?", "Item 8", k=1)] == ["net-income"]
