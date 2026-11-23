"""Offline graph, WebSocket and run-history behavior with fake upstreams."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from threading import Event, Lock

from bs4 import BeautifulSoup
import numpy as np
import pytest
from fastapi.testclient import TestClient

import agent
import data
from agent import AnalysisContext, app
from retrieval import CHUNKER_VERSION, MODEL_VERSION, LocalEmbedder, Passage, make_passages
from server import api
from storage import Store
from tests.support import FakeEmbedder, FakeGroq, FakeSEC, fake_vitals


def patch_finance(monkeypatch):
    monkeypatch.setattr(data, "getVitals", fake_vitals)
    monkeypatch.setattr(agent, "_insider_transactions", lambda ticker: [{"insider": "Example"}])


@pytest.mark.asyncio
async def test_graph_selects_filing_text_for_groq_and_reuses_passage_cache(tmp_path, monkeypatch):
    patch_finance(monkeypatch)
    store = Store(f"sqlite+aiosqlite:///{tmp_path / 'graph.db'}")
    await store.migrate()
    sec = FakeSEC()
    embedder = FakeEmbedder()
    groq = FakeGroq()
    with ThreadPoolExecutor(max_workers=3) as executor:
        ctx = AnalysisContext(sec, store, embedder, groq, executor, asyncio.Semaphore(2),
                              tuple(asyncio.Lock() for _ in range(64)))
        try:
            outputs = []
            async for update in app.astream({"ticker": "ACME"},
                                             config={"configurable": {"context": ctx}},
                                             stream_mode="updates"):
                outputs.append(next(iter(update)))
            assert outputs[0] == "fundamentals_agent"
            assert set(outputs[1:4]) == {"summary_agent", "risk_agent", "info_agent"}
            assert outputs[-1] == "synth_agent"
            result = await app.ainvoke({"ticker": "ACME"}, config={"configurable": {"context": ctx}})
            assert "## Financial Analysis" in result["final_output"]
            assert result["citations"][0]["excerpt"] in sec_source_text()
            assert all("Passages:" in prompt for prompt in groq.chat.completions.calls[:4])
            assert "industrial sensors for hospitals" in " ".join(groq.chat.completions.calls[:4])
            assert "priceHist" not in " ".join(groq.chat.completions.calls[:4])
            assert sec.fetches == 1
            assert sum(len(batch) > 1 for batch in embedder.calls) == 1
            sec.accession = "0000012345-27-000001"
            await app.ainvoke({"ticker": "ACME"}, config={"configurable": {"context": ctx}})
            assert sec.fetches == 2
            assert sum(len(batch) > 1 for batch in embedder.calls) == 2
        finally:
            await store.close()


@pytest.mark.asyncio
async def test_financial_summary_retrieves_distinct_income_and_balance_evidence(tmp_path):
    def vector(x, y, z):
        return np.array([x, y, z], dtype="<f4").tobytes()

    passages = [
        Passage("sales", "a", "Item 8", 0, "Consolidated net sales increased.", vector(1, 0, 0)),
        Passage("income", "a", "Item 8", 1, "Consolidated net income rose.", vector(0, 1, 0)),
        Passage("balance", "a", "Item 8", 2, "Cash and liabilities appear on the balance sheet.", vector(0, 0, 1)),
        Passage("unrelated", "a", "Item 8", 3, "Unrelated accounting note.", vector(-1, -1, -1)),
    ]

    class TopicEmbedder(LocalEmbedder):
        async def embed(self, texts):
            question = texts[0].lower()
            if "balance sheet" in question:
                return [vector(0, 0, 1)]
            if "net sales" in question:
                return [vector(1, 0, 0)]
            return [vector(0, 1, 0)]

    store = Store(f"sqlite+aiosqlite:///{tmp_path / 'query-test.db'}")
    groq = FakeGroq()
    with ThreadPoolExecutor(max_workers=1) as executor:
        ctx = AnalysisContext(FakeSEC(), store, TopicEmbedder(), groq, executor,
                              asyncio.Semaphore(2), tuple(asyncio.Lock() for _ in range(64)))
        try:
            result = await agent.summary_agent(
                {"passages": passages}, {"configurable": {"context": ctx}}
            )
            prompt = groq.chat.completions.calls[0]
            assert all(f"[{id_}]" in prompt for id_ in ("sales", "income", "balance"))
            assert "[unrelated]" not in prompt
            assert len(result["summary_claims"]) == 1
            assert result["summary_claims"][0]["citations"][0] in {"sales", "income", "balance"}
        finally:
            await store.close()


@pytest.mark.asyncio
async def test_evidence_selection_reserves_slot_for_later_questions_under_cap():
    questions = ("sales", "margin", "operating earnings", "outlook")
    passages = {
        question: [
            Passage(f"{question}-{rank}", "a", "Item 7", rank, question)
            for rank in range(2)
        ]
        for question in questions
    }

    class FocusedEmbedder:
        async def search(self, query, source, item, k):
            assert item == "Item 7" and k == 2
            return passages[query]

    selected = await agent.select_passages(FocusedEmbedder(), [], "Item 7", questions)
    assert list(selected) == [
        "sales-0", "margin-0", "operating earnings-0", "outlook-0",
        "sales-1", "margin-1",
    ]


@pytest.mark.asyncio
async def test_cached_old_parser_text_reindexes_without_refetch(tmp_path, monkeypatch):
    patch_finance(monkeypatch)
    source = (Path(__file__).parent / "fixtures" / "paged_filing.html").read_text()
    text = data._filing_text(BeautifulSoup(source, "html.parser"))
    store = Store(f"sqlite+aiosqlite:///{tmp_path / 'old-parser.db'}")
    await store.migrate()
    sec = FakeSEC()
    ref = await sec.resolve_filing("ACME")
    await store.save_filing(ref, text, {"Item 1": "truncated"}, "legacy")
    stale = [Passage(p.id, p.accession, p.item, p.index, p.text, b"\x00\x00\x80?") for p in
             make_passages(ref.accession, {"Item 1": "truncated"})]
    await store.save_passages(ref.accession, stale, MODEL_VERSION, CHUNKER_VERSION)
    with ThreadPoolExecutor(max_workers=3) as executor:
        ctx = AnalysisContext(sec, store, FakeEmbedder(), FakeGroq(), executor,
                              asyncio.Semaphore(2), tuple(asyncio.Lock() for _ in range(64)))
        try:
            result = await app.ainvoke({"ticker": "ACME"},
                                      config={"configurable": {"context": ctx}})
            assert "Financial Analysis" in result["final_output"]
            assert sec.fetches == 0
            assert (await store.get_filing(ref.accession))["parser_version"] == data.PARSER_VERSION
            business = (await store.get_filing(ref.accession))["sections"]["Item 1"]
            assert "First-page business" in business
            assert "Second-page business" in business
            cached = await store.get_passages(ref.accession, MODEL_VERSION, CHUNKER_VERSION)
            assert len(cached) == 4
            assert all("truncated" not in passage.text for passage in cached)
        finally:
            await store.close()


def sec_source_text():
    return " ".join(("Acme makes industrial sensors for hospitals.",
                     "A single supplier could interrupt production.",
                     "Revenue rose 8 percent in the fiscal year.",
                     "Revenue was 108 million dollars."))


def test_websocket_emits_stages_final_and_sql_history(tmp_path, monkeypatch):
    patch_finance(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-key")
    with TestClient(api) as client:
        api.state.context.sec = FakeSEC()
        api.state.context.embedder = FakeEmbedder()
        api.state.context.groq = FakeGroq()
        with client.websocket_connect("/test") as websocket:
            websocket.send_json({"ticker": "ACME"})
            events = []
            while True:
                event = websocket.receive_json()
                events.append(event)
                if event["type"] in {"final", "error"}:
                    break
        assert [event["stage"] for event in events if event["type"] == "stage"][0] == "fundamentals"
        assert events[-1]["type"] == "final", events[-1]
        run = events[-1]["run"]
        assert run["status"] == "completed" and run["accession"]
        assert run["citations"] and run["citations"][0]["url"].startswith("https://www.sec.gov/")
        assert client.get("/runs").json()[0]["id"] == run["id"]
        assert client.get(f"/runs/{run['id']}").json()["finalBrief"] == run["finalBrief"]
        assert client.get("/runs/missing").status_code == 404


def test_websocket_rejects_invalid_ticker_and_persists_upstream_error(tmp_path, monkeypatch):
    patch_finance(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'errors.db'}")
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-key")
    with TestClient(api) as client:
        api.state.context.sec = FakeSEC()
        api.state.context.embedder = FakeEmbedder()
        api.state.context.groq = FakeGroq(invalid_citation=True)
        with client.websocket_connect("/test") as websocket:
            websocket.send_json({"ticker": "@@@"})
            rejected = websocket.receive_json()
        assert rejected["type"] == "error" and "ticker" in rejected["message"]
        assert client.get("/runs").json() == []
        with client.websocket_connect("/test") as websocket:
            websocket.send_json({"ticker": "ACME"})
            while True:
                failed = websocket.receive_json()
                if failed["type"] == "error":
                    break
        assert failed["run"]["status"] == "failed"
        assert failed["run"]["error"] == "ValueError"
        assert client.get("/runs").json()[0]["id"] == failed["run"]["id"]


def test_two_websockets_finish_independently_and_persist_history(tmp_path, monkeypatch):
    patch_finance(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sockets.db'}")
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-key")
    first_fetch_started = Event()
    release_first = Event()

    class PausedFirstSEC(FakeSEC):
        def __init__(self):
            super().__init__()
            self._first_fetch = True
            self._lock = Lock()

        async def fetch_filing(self, ref):
            with self._lock:
                first = self._first_fetch
                self._first_fetch = False
            if first:
                first_fetch_started.set()
                if not await asyncio.to_thread(release_first.wait, 30):
                    raise TimeoutError("First SEC fetch was not released")
            return await super().fetch_filing(ref)

    with TestClient(api) as client:
        api.state.context.sec = PausedFirstSEC()
        api.state.context.embedder = FakeEmbedder()
        api.state.context.groq = FakeGroq()

        def terminal_event(websocket):
            while True:
                event = websocket.receive_json()
                if event["type"] in {"final", "error"}:
                    return event

        with ThreadPoolExecutor(max_workers=2) as workers:
            with client.websocket_connect("/test") as first_socket:
                try:
                    first_socket.send_json({"ticker": "ACME"})
                    first = workers.submit(terminal_event, first_socket)
                    assert first_fetch_started.wait(timeout=5), "First WebSocket never reached SEC fetch"

                    with client.websocket_connect("/test") as second_socket:
                        try:
                            second_socket.send_json({"ticker": "ACME"})
                            second = workers.submit(terminal_event, second_socket)
                            try:
                                second_event = second.result(timeout=10)
                            except FutureTimeoutError:
                                pytest.fail("Second WebSocket did not finish while first SEC fetch was held")
                            assert not release_first.is_set()
                            assert not first.done()
                            assert second_event["type"] == "final", second_event
                            second_run = second_event["run"]
                            assert second_run["status"] == "completed"
                            assert "Financial Analysis" in second_run["finalBrief"]

                            history = client.get("/runs").json()
                            assert len(history) == 2
                            assert {run["status"] for run in history} == {"running", "completed"}
                            assert next(run for run in history if run["status"] == "completed")["id"] == second_run["id"]
                            first_id = next(run for run in history if run["status"] == "running")["id"]
                            assert first_id != second_run["id"]
                            assert not first.done()

                            release_first.set()
                            first_event = first.result(timeout=10)
                            assert first_event["type"] == "final", first_event
                            first_run = first_event["run"]
                            assert first_run["id"] == first_id
                            assert first_run["status"] == "completed"

                            completed = client.get("/runs").json()
                            assert {run["id"] for run in completed} == {first_run["id"], second_run["id"]}
                            assert all(run["status"] == "completed" for run in completed)
                            for run in (first_run, second_run):
                                saved = client.get(f"/runs/{run['id']}").json()
                                assert saved["status"] == "completed"
                                assert saved["finalBrief"] == run["finalBrief"]
                                assert saved["citations"] == run["citations"]
                        finally:
                            release_first.set()
                finally:
                    release_first.set()


@pytest.mark.asyncio
async def test_waiting_upstream_call_does_not_block_another_analysis(tmp_path, monkeypatch):
    patch_finance(monkeypatch)
    first_fetch_started = asyncio.Event()
    release_first = asyncio.Event()

    class SlowFirstSEC(FakeSEC):
        async def fetch_filing(self, ref):
            if not first_fetch_started.is_set():
                first_fetch_started.set()
                await release_first.wait()
            return await super().fetch_filing(ref)

    store = Store(f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}")
    await store.migrate()
    with ThreadPoolExecutor(max_workers=3) as executor:
        ctx = AnalysisContext(SlowFirstSEC(), store, FakeEmbedder(), FakeGroq(),
                              executor, asyncio.Semaphore(2),
                              tuple(asyncio.Lock() for _ in range(64)))
        config = {"configurable": {"context": ctx}}
        first = asyncio.create_task(app.ainvoke({"ticker": "ACME"}, config=config))
        try:
            await asyncio.wait_for(first_fetch_started.wait(), timeout=2)
            second = await asyncio.wait_for(app.ainvoke({"ticker": "ACME"}, config=config),
                                            timeout=2)
            assert "Financial Analysis" in second["final_output"]
            assert not first.done()
            release_first.set()
            assert "Financial Analysis" in (await asyncio.wait_for(first, timeout=2))["final_output"]
        finally:
            release_first.set()
            await asyncio.gather(first, return_exceptions=True)
            await store.close()


def test_client_disconnect_records_failed_run(tmp_path, monkeypatch):
    import time

    patch_finance(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'disconnect.db'}")
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-key")

    class WaitingSEC(FakeSEC):
        async def fetch_filing(self, ref):
            await asyncio.sleep(30)
            return await super().fetch_filing(ref)

    with TestClient(api) as client:
        api.state.context.sec = WaitingSEC()
        api.state.context.embedder = FakeEmbedder()
        api.state.context.groq = FakeGroq()
        with client.websocket_connect("/test") as websocket:
            websocket.send_json({"ticker": "ACME"})
            for _ in range(100):
                if client.get("/runs").json():
                    break
                time.sleep(0.01)
        for _ in range(100):
            runs = client.get("/runs").json()
            if runs and runs[0]["status"] == "failed":
                assert runs[0]["error"] == "Client disconnected"
                break
            time.sleep(0.01)
        else:
            pytest.fail("Disconnected analysis did not become a failed run")
