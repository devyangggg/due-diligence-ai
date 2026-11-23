"""Isolated SQLite schema, persistence, and accession cache contract tests."""

from dataclasses import replace

import pytest

from data import FilingRef, PARSER_VERSION
from retrieval import CHUNKER_VERSION, MODEL_VERSION, make_passages
from storage import Store


@pytest.mark.asyncio
async def test_run_and_filing_cache_survive_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}"
    ref = FilingRef("ACME", "0000012345", "0000012345-26-000001", "2026-02-01", "https://www.sec.gov/example.htm")
    first = Store(url)
    await first.migrate()
    await first.migrate()
    await first.save_filing(ref, "Item 1. Business\nSensors", {"Item 1": "Sensors"},
                            PARSER_VERSION)
    passages = [replace(passage, embedding=b"\x00\x00\x80?") for passage in
                make_passages(ref.accession, {"Item 1": "Sensors"})]
    await first.save_passages(ref.accession, passages, MODEL_VERSION, CHUNKER_VERSION)
    run = await first.create_run("ACME")
    citation = {"id": passages[0].id, "item": "Item 1", "url": ref.url, "excerpt": "Sensors"}
    await first.update_run(run["id"], status="completed", accession=ref.accession,
                           brief="Sensors are sold [source](https://www.sec.gov/example.htm)", citations=[citation])
    failed = await first.create_run("BAD")
    await first.update_run(failed["id"], status="failed", error="upstream_timeout")
    await first.close()

    second = Store(url)
    try:
        await second.migrate()
        assert (await second.get_filing(ref.accession))["sections"] == {"Item 1": "Sensors"}
        assert await second.get_passages(ref.accession, MODEL_VERSION, CHUNKER_VERSION) == passages
        assert await second.get_passages(ref.accession, MODEL_VERSION + "-new", CHUNKER_VERSION) == []
        restored = await second.get_run(run["id"])
        assert restored["status"] == "completed"
        assert restored["citations"] == [citation]
        assert restored["finalBrief"].startswith("Sensors")
        assert restored["finishedAt"]
        assert (await second.get_run(failed["id"]))["error"] == "upstream_timeout"
        assert len(await second.list_runs()) == 2
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_passage_cache_versions_do_not_overwrite_each_other(tmp_path):
    store = Store(f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}")
    try:
        await store.migrate()
        ref = FilingRef("ACME", "0000012345", "a", "2026-02-01", "https://www.sec.gov/example.htm")
        await store.save_filing(ref, "body", {"Item 1": "body"}, PARSER_VERSION)
        old = [replace(p, embedding=b"one") for p in make_passages("a", {"Item 1": "body"})]
        new = [replace(p, embedding=b"two") for p in old]
        await store.save_passages("a", old, "model-v1", "chunk-v1")
        await store.save_passages("a", new, "model-v2", "chunk-v1")
        assert await store.get_passages("a", "model-v1", "chunk-v1") == old
        assert await store.get_passages("a", "model-v2", "chunk-v1") == new
        await store.save_passages("a", new, "model-v1", "chunk-v1")
        assert await store.get_passages("a", "model-v1", "chunk-v1") == old
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_unfinished_run_is_marked_interrupted_after_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'interrupted.db'}"
    first = Store(url)
    await first.migrate()
    run = await first.create_run("ACME")
    await first.close()
    second = Store(url)
    try:
        await second.migrate()
        interrupted = await second.get_run(run["id"])
        assert interrupted["status"] == "interrupted"
        assert interrupted["finishedAt"]
        assert interrupted["error"] == "Backend restarted before completion"
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_parser_upgrade_invalidates_cached_passages(tmp_path):
    store = Store(f"sqlite+aiosqlite:///{tmp_path / 'parser.db'}")
    try:
        await store.migrate()
        ref = FilingRef("ACME", "0000012345", "a", "2026-02-01",
                        "https://www.sec.gov/example.htm")
        await store.save_filing(ref, "full source", {"Item 1": "wrong old slice"}, "legacy")
        old = [replace(p, embedding=b"one") for p in
               make_passages("a", {"Item 1": "wrong old slice"})]
        await store.save_passages("a", old, MODEL_VERSION, CHUNKER_VERSION)
        await store.update_sections("a", {"Item 1": "correct complete slice"}, PARSER_VERSION)
        cached = await store.get_filing("a")
        assert cached["parser_version"] == PARSER_VERSION
        assert cached["sections"]["Item 1"] == "correct complete slice"
        assert await store.get_passages("a", MODEL_VERSION, CHUNKER_VERSION) == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migrates_existing_v1_database_without_discarding_filings(tmp_path):
    import sqlite3

    path = tmp_path / "previous.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE filings (
                accession TEXT PRIMARY KEY, cik TEXT, ticker TEXT, filing_date TEXT,
                url TEXT, text TEXT, sections TEXT
            );
            CREATE TABLE passages (
                accession TEXT, model_version TEXT, chunker_version TEXT, item TEXT,
                passage_index INTEGER, id TEXT, text TEXT, embedding BLOB
            );
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, ticker TEXT, accession TEXT, status TEXT,
                started_at TEXT, finished_at TEXT, final_brief TEXT,
                citations TEXT DEFAULT '[]', error TEXT
            );
            INSERT INTO filings VALUES (
                'a', '0000012345', 'ACME', '2026-02-01', 'https://www.sec.gov/example.htm',
                'full source', '{"Item 1":"old slice"}'
            );
            PRAGMA user_version=1;
        """)
    store = Store(f"sqlite+aiosqlite:///{path}")
    try:
        await store.migrate()
        assert (await store.get_filing("a"))["parser_version"] == "legacy"
        await store.migrate()
        await store.update_sections("a", {"Item 1": "reparsed"}, PARSER_VERSION)
        assert (await store.get_filing("a"))["sections"]["Item 1"] == "reparsed"
    finally:
        await store.close()
