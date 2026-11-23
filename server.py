"""FastAPI WebSocket analysis, run history, and lifecycle-managed dependencies."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from dotenv import load_dotenv

from agent import AnalysisContext, app
from data import SECClient
from retrieval import LocalEmbedder
from storage import Store

logger = logging.getLogger(__name__)
TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}\Z")


@asynccontextmanager
async def lifespan(api: FastAPI):
    load_dotenv()
    http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=True)
    store = Store(os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./due_diligence.db"))
    executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="finance")
    try:
        await store.migrate()
        key = os.getenv("GROQ_API_KEY")
        api.state.context = AnalysisContext(
            sec=SECClient(http, user_agent=os.getenv("SEC_USER_AGENT", "")),
            store=store,
            embedder=LocalEmbedder(),
            groq=AsyncGroq(api_key=key, max_retries=2, timeout=60.0) if key else None,
            executor=executor,
            model_limit=asyncio.Semaphore(2),
            index_locks=tuple(asyncio.Lock() for _ in range(64)),
        )
        yield
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        await store.close()
        await http.aclose()


api = FastAPI(lifespan=lifespan)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@api.get("/runs")
async def list_runs():
    return await api.state.context.store.list_runs(limit=20)


@api.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = await api.state.context.store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


async def _watch_disconnect(websocket: WebSocket):
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


async def _execute(websocket: WebSocket, ticker: str, run_id: str, ctx: AnalysisContext):
    final = None
    citations = []
    accession = None
    async for updates in app.astream(
        {"ticker": ticker}, config={"configurable": {"context": ctx}}, stream_mode="updates"
    ):
        for node, values in updates.items():
            stage = node.removesuffix("_agent")
            if node == "fundamentals_agent":
                accession = values["filing_ref"].accession
                await ctx.store.update_run(run_id, status="running", accession=accession)
            if node == "synth_agent":
                final = values["final_output"]
                citations = values["citations"]
            await websocket.send_json({"type": "stage", "stage": stage, "status": "completed"})
    if not final:
        raise ValueError("Analysis finished without a final brief")
    return final, citations, accession


@api.websocket("/test")
async def eval_model(websocket: WebSocket):
    await websocket.accept()
    ctx: AnalysisContext = api.state.context
    try:
        payload = await websocket.receive_json()
        if not isinstance(payload, dict) or not isinstance(payload.get("ticker"), str):
            raise ValueError("Send JSON with a ticker string")
        ticker = payload["ticker"].strip().upper()
        if not TICKER_PATTERN.fullmatch(ticker):
            raise ValueError("Enter a valid ticker (1–10 letters, digits, dots, or hyphens)")
    except (ValueError, WebSocketDisconnect) as exc:
        if not isinstance(exc, WebSocketDisconnect):
            await websocket.send_json({"type": "error", "run": None, "message": str(exc)})
            await websocket.close()
        return
    if ctx.groq is None:
        await websocket.send_json({
            "type": "error", "run": None, "message": "GROQ_API_KEY is not configured"
        })
        await websocket.close()
        return

    run = await ctx.store.create_run(ticker)
    started = time.monotonic()
    task = asyncio.create_task(_execute(websocket, ticker, run["id"], ctx))
    watcher = asyncio.create_task(_watch_disconnect(websocket))
    try:
        done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done and task not in done:
            with anyio.CancelScope(shield=True):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await ctx.store.update_run(run["id"], status="failed", error="Client disconnected")
            logger.info("run_disconnected id=%s ticker=%s", run["id"], ticker)
            return
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        brief, citations, accession = task.result()
        completed = await ctx.store.update_run(
            run["id"], status="completed", accession=accession,
            brief=brief, citations=citations
        )
        logger.info("run_completed id=%s ticker=%s seconds=%.3f", run["id"], ticker,
                    time.monotonic() - started)
        await websocket.send_json({"type": "final", "run": completed})
        await websocket.close()
    except (Exception, asyncio.CancelledError) as exc:
        with anyio.CancelScope(shield=True):
            task.cancel()
            watcher.cancel()
            await asyncio.gather(task, watcher, return_exceptions=True)
            failed = await ctx.store.update_run(
                run["id"], status="failed", error=type(exc).__name__
            )
            logger.exception("run_failed id=%s ticker=%s seconds=%.3f", run["id"], ticker,
                             time.monotonic() - started)
            try:
                await websocket.send_json({
                    "type": "error", "run": failed,
                    "message": f"Analysis failed (run {run['id']}); see server logs",
                })
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass
        if isinstance(exc, asyncio.CancelledError):
            raise
