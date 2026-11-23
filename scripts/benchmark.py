"""Time the current fake-backed graph with shared offline inputs and model cost.

Compare with scripts.legacy_baseline as controlled mock workflows only: the
historical graph has no cache, while this graph uses a warm SQLite cache.
Run from repository root: uv run python -m scripts.benchmark --label local-change
"""

import argparse
import asyncio
import json
import platform
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import agent
import data
from agent import AnalysisContext, app
from storage import Store
from scripts.benchmark_fixture import (
    INSIDER_TRANSACTIONS, MODEL_CALL_LATENCY_SECONDS, TICKER,
    FixtureSEC, LatencyLLM, fake_vitals,
)
from tests.support import FakeEmbedder


def percentile(values, percentage):
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * percentage + 99) // 100 - 1)]


async def measure(label, count):
    data.getVitals = fake_vitals
    agent._insider_transactions = lambda ticker: INSIDER_TRANSACTIONS.copy()
    with tempfile.TemporaryDirectory(prefix="diligence-benchmark-") as directory:
        store = Store(f"sqlite+aiosqlite:///{Path(directory) / 'benchmark.db'}")
        await store.migrate()
        sec = FixtureSEC()
        embedder = FakeEmbedder()
        llm = LatencyLLM()
        with ThreadPoolExecutor(max_workers=3) as executor:
            ctx = AnalysisContext(sec, store, embedder, llm, executor,
                                  asyncio.Semaphore(2), tuple(asyncio.Lock() for _ in range(64)))
            runs = []
            try:
                for index in range(count):
                    start = time.monotonic()
                    last = start
                    stages = {}
                    async for update in app.astream({"ticker": TICKER},
                                                    config={"configurable": {"context": ctx}},
                                                    stream_mode="updates"):
                        now = time.monotonic()
                        node = next(iter(update))
                        stages[node] = now - last
                        last = now
                    runs.append({"number": index + 1, "cache": "cold" if index == 0 else "warm",
                                 "durationSeconds": time.monotonic() - start,
                                 "stageSeconds": stages})
            finally:
                await store.close()
    durations = [run["durationSeconds"] for run in runs]
    warm = durations[1:]
    return {
        "label": label,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "data": "shared synthetic SEC item text and Yahoo vitals/insider inputs; no external calls",
        "modelResponse": "citation-bearing JSON; historical graph uses plain-text summaries",
        "modelCallLatencySeconds": MODEL_CALL_LATENCY_SECONDS,
        "cacheState": "SQLite: first run cold, remaining runs warm; historical graph has no cache",
        "comparison": "controlled mock workflow comparison only; not a production speedup",
        "runCount": count, "fetchCount": sec.fetches, "embeddingCalls": len(embedder.calls),
        "modelCalls": llm.chat.completions.calls_with_latency,
        "medianSeconds": statistics.median(durations),
        "p95Seconds": percentile(durations, 95),
        "warmMedianSeconds": statistics.median(warm),
        "warmP95Seconds": percentile(warm, 95),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="revision or scenario name")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(measure(args.label, 21)), indent=2))


if __name__ == "__main__":
    main()
