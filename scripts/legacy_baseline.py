"""Time the pinned historical graph with the shared offline benchmark inputs.

This controlled mock workflow comparison is not a production speedup estimate.
Run: uv run python -m scripts.legacy_baseline > /tmp/diligence-legacy.json
"""

import json
import platform
import statistics
import subprocess
import time
from types import SimpleNamespace
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from scripts.benchmark_fixture import (
    FILING_TEXT, INSIDER_TRANSACTIONS, MODEL_CALL_LATENCY_SECONDS,
    SECTIONS, TICKER, fake_vitals, legacy_model_delay,
)

BASELINE_COMMIT = "1d4b036045f9afa85e1d0086a02f345a7c6059a3"


class LegacyGroq:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        self.calls += 1
        legacy_model_delay()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Synthetic summary from mocked response"))])


def percentile(values, percentage):
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * percentage + 99) // 100 - 1)]


def initial_state():
    return {"ticker": TICKER, "fundamentals": {}, "risk_info": {},
            "info_collected": {}, "all_agent_done": [], "final_output": ""}


def main():
    old = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:agent.py"], text=True)
    start = old.index("class AgentState(TypedDict):")
    end = old.index("\n# result = app.invoke", start)
    sections = SECTIONS.copy()
    legacy_data = SimpleNamespace(getVitals=fake_vitals,
                                  get_latest_10k=lambda ticker: FILING_TEXT,
                                  slice_10k_specific_items=lambda text: sections)
    legacy_fin = SimpleNamespace(Ticker=lambda ticker: SimpleNamespace(
        insider_transactions=pd.DataFrame(INSIDER_TRANSACTIONS)))
    model = LegacyGroq()
    namespace = {"TypedDict": TypedDict, "data": legacy_data, "fin": legacy_fin,
                 "pd": pd, "StateGraph": StateGraph, "START": START, "END": END,
                 "client": model}
    exec(compile(old[start:end], "legacy-agent.py", "exec"), namespace)
    runs = []
    for number in range(21):
        started = time.monotonic()
        previous = started
        stages = {}
        for update in namespace["app"].stream(initial_state()):
            now = time.monotonic()
            stages[next(iter(update))] = now - previous
            previous = now
        runs.append({"number": number + 1, "cache": "uncached",
                     "durationSeconds": time.monotonic() - started,
                     "stageSeconds": stages})
    assert model.calls == 5 * len(runs)
    sections.pop("Item 8")
    try:
        list(namespace["app"].stream(initial_state()))
    except KeyError as exc:
        missing_section_failure = str(exc)
    else:
        raise AssertionError("Historical graph unexpectedly tolerated a missing item")
    durations = [run["durationSeconds"] for run in runs]
    print(json.dumps({
        "source": f"historical {BASELINE_COMMIT}:agent.py",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "data": "shared synthetic SEC item text and Yahoo vitals/insider inputs; no external calls",
        "modelResponse": "historical plain-text summary; current graph uses citation-bearing JSON",
        "modelCallLatencySeconds": MODEL_CALL_LATENCY_SECONDS,
        "cacheState": "no historical cache; all 21 runs are uncached (first is not a cold-cache run)",
        "comparison": "controlled mock workflow comparison only; not a production speedup",
        "runCount": len(runs), "groqCallsPerRun": 5,
        "missingSectionFailure": missing_section_failure,
        "medianSeconds": statistics.median(durations),
        "p95Seconds": percentile(durations, 95), "runs": runs,
    }, indent=2))


if __name__ == "__main__":
    main()
