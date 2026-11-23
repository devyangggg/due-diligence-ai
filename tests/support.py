"""Deterministic, network-free collaborators for the analysis graph and API."""

import json
import re
from types import SimpleNamespace

import numpy as np

from data import Filing, FilingRef
from retrieval import LocalEmbedder


class FakeSEC:
    def __init__(self, accession="0000012345-26-000001"):
        self.accession = accession
        self.fetches = 0
        self.resolves = 0

    async def resolve_filing(self, ticker):
        self.resolves += 1
        return FilingRef(ticker=ticker, cik="0000012345", accession=self.accession,
                         filing_date="2026-02-01", url="https://www.sec.gov/Archives/edgar/data/12345/example.htm")

    async def fetch_filing(self, ref):
        self.fetches += 1
        sections = {
            "Item 1": "Acme makes industrial sensors for hospitals.",
            "Item 1A": "A single supplier could interrupt production.",
            "Item 7": "Revenue rose 8 percent in the fiscal year.",
            "Item 8": "Revenue was 108 million dollars.",
        }
        return Filing(ref, "\n\n".join(sections.values()), sections)


class FakeEmbedder(LocalEmbedder):
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(tuple(texts))
        return [np.array([1., 0.], dtype="<f4").tobytes() for _ in texts]


class FakeCompletions:
    def __init__(self, invalid_citation=False):
        self.calls = []
        self.invalid_citation = invalid_citation

    async def create(self, *, messages, **kwargs):
        user = messages[-1]["content"]
        self.calls.append(user)
        if user.startswith("Item:"):
            identifier = re.search(r"\[([^\]]+)]", user).group(1)
            if self.invalid_citation:
                identifier = "invented-passage"
            content = json.dumps({"claims": [{"text": "A stated filing fact.",
                                               "citations": [identifier]}]})
        else:
            data = json.loads(user)
            identifier = data["filing_claims"][0]["citations"][0]
            headings = ["Company Snapshot", "Financial Analysis", "Risk Assessment",
                        "Insider Activity", "Bull Case", "Bear Case", "Recommendation"]
            content = json.dumps({"sections": [
                {"heading": heading, "claims": [{"text": "A stated filing fact.",
                 "citations": [identifier]}]} if heading == "Financial Analysis" else
                {"heading": heading, "claims": [{"text": "Analysis from supplied inputs.",
                 "citations": []}]} for heading in headings
            ]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeGroq:
    def __init__(self, invalid_citation=False):
        self.chat = SimpleNamespace(completions=FakeCompletions(invalid_citation))


def fake_vitals(_ticker):
    return {"peRatio": 12.0, "sector": "Industrials", "marketCap": 1000000,
            "priceHist": [100., 80., 120.], "dailyReturns": [1., -2., 3.], "beta": 1.2}
