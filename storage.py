"""SQLite filing, embedding, and run history storage for the local analysis app."""

import json
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import event, text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine


_SCHEMA_VERSION = 2
_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted", "cancelled"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_dict(row) -> dict:
    return {
        "id": row["id"],
        "ticker": row["ticker"],
        "accession": row["accession"],
        "status": row["status"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
        "finalBrief": row["final_brief"],
        "citations": json.loads(row["citations"]),
        "error": row["error"],
    }


class Store:
    """Single-user, local SQLite store. Call migrate() before accessing data."""

    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url, connect_args={"timeout": 30})

        @event.listens_for(self._engine.sync_engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
            finally:
                cursor.close()

    async def migrate(self) -> None:
        async with self._engine.connect() as connection:
            # WAL persists for file databases and permits readers during writes.
            await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            try:
                # Serialize migration across Store instances and processes.
                await connection.exec_driver_sql("BEGIN IMMEDIATE")
                version = (await connection.exec_driver_sql("PRAGMA user_version")).scalar_one()
                if version > _SCHEMA_VERSION:
                    raise RuntimeError(f"SQLite schema version {version} is newer than supported {_SCHEMA_VERSION}")
                if version < 1:
                    await connection.exec_driver_sql("""
                        CREATE TABLE IF NOT EXISTS filings (
                            accession TEXT PRIMARY KEY,
                            cik TEXT NOT NULL,
                            ticker TEXT NOT NULL,
                            filing_date TEXT NOT NULL,
                            url TEXT NOT NULL,
                            text TEXT NOT NULL,
                            sections TEXT NOT NULL
                        )
                    """)
                    await connection.exec_driver_sql("""
                        CREATE TABLE IF NOT EXISTS passages (
                            accession TEXT NOT NULL REFERENCES filings(accession) ON DELETE CASCADE,
                            model_version TEXT NOT NULL,
                            chunker_version TEXT NOT NULL,
                            item TEXT NOT NULL,
                            passage_index INTEGER NOT NULL,
                            id TEXT NOT NULL,
                            text TEXT NOT NULL,
                            embedding BLOB,
                            PRIMARY KEY (accession, model_version, chunker_version, item, passage_index)
                        )
                    """)
                    await connection.exec_driver_sql("""
                        CREATE TABLE IF NOT EXISTS runs (
                            id TEXT PRIMARY KEY,
                            ticker TEXT NOT NULL,
                            accession TEXT,
                            status TEXT NOT NULL,
                            started_at TEXT NOT NULL,
                            finished_at TEXT,
                            final_brief TEXT,
                            citations TEXT NOT NULL DEFAULT '[]',
                            error TEXT
                        )
                    """)
                    await connection.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs(started_at DESC, id DESC)"
                    )
                    await connection.exec_driver_sql("PRAGMA user_version=1")
                if version < 2:
                    await connection.exec_driver_sql(
                        "ALTER TABLE filings ADD COLUMN parser_version TEXT NOT NULL DEFAULT 'legacy'"
                    )
                    await connection.exec_driver_sql("PRAGMA user_version=2")
                # An ungraceful restart cannot leave an apparently active run forever.
                await connection.execute(sql_text("""
                    UPDATE runs SET status = 'interrupted', finished_at = :finished_at,
                                    error = 'Backend restarted before completion'
                    WHERE status = 'running'
                """), {"finished_at": _utc_now()})
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def close(self) -> None:
        await self._engine.dispose()

    async def get_filing(self, accession: str) -> dict | None:
        async with self._engine.connect() as connection:
            row = (await connection.execute(
                sql_text("SELECT accession, cik, ticker, filing_date, url, text, sections, "
                     "parser_version FROM filings WHERE accession = :accession"),
                {"accession": accession},
            )).mappings().one_or_none()
        if row is None:
            return None
        filing = dict(row)
        filing["sections"] = json.loads(filing["sections"])
        return filing

    async def save_filing(self, ref, text: str, sections: dict, parser_version: str) -> None:
        filing_date = ref.filing_date
        if hasattr(filing_date, "isoformat"):
            filing_date = filing_date.isoformat()
        async with self._engine.begin() as connection:
            # A later concurrent fetch of the same immutable SEC accession cannot overwrite its cache.
            await connection.execute(sql_text("""
                INSERT INTO filings
                    (accession, cik, ticker, filing_date, url, text, sections, parser_version)
                VALUES
                    (:accession, :cik, :ticker, :filing_date, :url, :text, :sections, :parser_version)
                ON CONFLICT(accession) DO NOTHING
            """), {
                "accession": ref.accession,
                "cik": str(ref.cik),
                "ticker": ref.ticker,
                "filing_date": filing_date,
                "url": ref.url,
                "text": text,
                "sections": json.dumps(sections),
                "parser_version": parser_version,
            })

    async def update_sections(self, accession: str, sections: dict, parser_version: str) -> None:
        """Reparse cached source once on parser upgrades and invalidate derived vectors."""
        async with self._engine.begin() as connection:
            result = await connection.execute(sql_text("""
                UPDATE filings SET sections = :sections, parser_version = :parser_version
                WHERE accession = :accession
            """), {"accession": accession, "sections": json.dumps(sections),
                   "parser_version": parser_version})
            if not result.rowcount:
                raise KeyError(accession)
            await connection.execute(sql_text(
                "DELETE FROM passages WHERE accession = :accession"
            ), {"accession": accession})

    async def get_passages(self, accession: str, model_version: str, chunker_version: str) -> list["Passage"]:
        from retrieval import Passage

        async with self._engine.connect() as connection:
            rows = (await connection.execute(sql_text("""
                SELECT id, accession, item, passage_index, text, embedding
                FROM passages
                WHERE accession = :accession AND model_version = :model_version
                  AND chunker_version = :chunker_version
                ORDER BY item, passage_index
            """), {
                "accession": accession,
                "model_version": model_version,
                "chunker_version": chunker_version,
            })).mappings().all()
        return [Passage(id=row["id"], accession=row["accession"], item=row["item"],
                        index=row["passage_index"], text=row["text"], embedding=row["embedding"])
                for row in rows]

    async def save_passages(self, accession: str, passages: list["Passage"],
                            model_version: str, chunker_version: str) -> None:
        if not passages:
            return
        keys = set()
        ids = set()
        for passage in passages:
            if passage.accession != accession:
                raise ValueError("Passage accession does not match cache accession")
            key = (passage.item, passage.index)
            if key in keys:
                raise ValueError(f"Duplicate passage item/index: {key!r}")
            if passage.id in ids:
                raise ValueError(f"Duplicate passage id: {passage.id!r}")
            keys.add(key)
            ids.add(passage.id)

        async with self._engine.connect() as connection:
            try:
                # Reserve the writer slot before checking the cache: two writers cannot each
                # observe a miss and then publish partial or competing passage sets.
                await connection.exec_driver_sql("BEGIN IMMEDIATE")
                existing = (await connection.execute(sql_text("""
                    SELECT COUNT(*) AS total, COUNT(embedding) AS embedded
                    FROM passages
                    WHERE accession = :accession AND model_version = :model_version
                      AND chunker_version = :chunker_version
                """), {
                    "accession": accession,
                    "model_version": model_version,
                    "chunker_version": chunker_version,
                })).mappings().one()
                if existing["total"] and existing["embedded"] == existing["total"]:
                    await connection.commit()
                    return

                await connection.execute(sql_text("""
                    DELETE FROM passages
                    WHERE accession = :accession AND model_version = :model_version
                      AND chunker_version = :chunker_version
                """), {
                    "accession": accession,
                    "model_version": model_version,
                    "chunker_version": chunker_version,
                })
                await connection.execute(sql_text("""
                    INSERT INTO passages
                        (accession, model_version, chunker_version, item, passage_index, id, text, embedding)
                    VALUES
                        (:accession, :model_version, :chunker_version, :item, :passage_index, :id, :text, :embedding)
                """), [{
                    "accession": accession,
                    "model_version": model_version,
                    "chunker_version": chunker_version,
                    "item": passage.item,
                    "passage_index": passage.index,
                    "id": passage.id,
                    "text": passage.text,
                    "embedding": passage.embedding,
                } for passage in passages])
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def create_run(self, ticker: str) -> dict:
        run_id = str(uuid4())
        started_at = _utc_now()
        async with self._engine.begin() as connection:
            await connection.execute(sql_text("""
                INSERT INTO runs (id, ticker, status, started_at)
                VALUES (:id, :ticker, 'running', :started_at)
            """), {"id": run_id, "ticker": ticker, "started_at": started_at})
        return {
            "id": run_id, "ticker": ticker, "accession": None, "status": "running",
            "startedAt": started_at, "finishedAt": None, "finalBrief": None,
            "citations": [], "error": None,
        }

    async def update_run(self, run_id: str, *, status: str, accession: str | None = None,
                         brief: str | None = None, citations: list[dict] | None = None,
                         error: str | None = None) -> dict:
        values = {"status": status}
        if status in _TERMINAL_STATUSES:
            values["finished_at"] = _utc_now()
        if accession is not None:
            values["accession"] = accession
        if brief is not None:
            values["final_brief"] = brief
        if citations is not None:
            values["citations"] = json.dumps(citations)
        if error is not None:
            values["error"] = error
        updates = ", ".join(f"{column} = :{column}" for column in values)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                sql_text(f"UPDATE runs SET {updates} WHERE id = :run_id"),
                {"run_id": run_id, **values},
            )
            if not result.rowcount:
                raise KeyError(run_id)
            row = (await connection.execute(
                sql_text("SELECT * FROM runs WHERE id = :run_id"), {"run_id": run_id}
            )).mappings().one()
            return _run_dict(row)

    async def get_run(self, run_id: str) -> dict | None:
        async with self._engine.connect() as connection:
            row = (await connection.execute(
                sql_text("SELECT * FROM runs WHERE id = :run_id"), {"run_id": run_id}
            )).mappings().one_or_none()
        return _run_dict(row) if row is not None else None

    async def list_runs(self, limit: int = 20) -> list[dict]:
        async with self._engine.connect() as connection:
            rows = (await connection.execute(sql_text("""
                SELECT * FROM runs ORDER BY started_at DESC, id DESC LIMIT :limit
            """), {"limit": max(0, min(limit, 100))})).mappings().all()
        return [_run_dict(row) for row in rows]
