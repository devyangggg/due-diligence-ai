"""Shared offline inputs and per-completion cost for the two graph harnesses."""

import asyncio
import time

from data import Filing
from tests.support import FakeGroq, FakeSEC, fake_vitals

TICKER = "ACME"
MODEL_CALL_LATENCY_SECONDS = 0.002
SECTIONS = {
    "Item 1": "Acme makes industrial sensors for hospitals.",
    "Item 1A": "A single supplier could interrupt production.",
    "Item 7": "Revenue rose 8 percent in the fiscal year.",
    "Item 8": "Revenue was 108 million dollars.",
}
FILING_TEXT = "\n\n".join(SECTIONS.values())
INSIDER_TRANSACTIONS = [{"Insider": "Synthetic"}]


class FixtureSEC(FakeSEC):
    async def fetch_filing(self, ref):
        self.fetches += 1
        return Filing(ref, FILING_TEXT, SECTIONS.copy())


class LatencyGroq(FakeGroq):
    def __init__(self):
        super().__init__()
        original_create = self.chat.completions.create
        completions = self.chat.completions
        completions.calls_with_latency = 0

        async def create(**kwargs):
            completions.calls_with_latency += 1
            await asyncio.sleep(MODEL_CALL_LATENCY_SECONDS)
            return await original_create(**kwargs)

        completions.create = create


def legacy_model_delay():
    time.sleep(MODEL_CALL_LATENCY_SECONDS)
