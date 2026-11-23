# Due-diligence AI

A single-user local application that builds a source-linked due-diligence brief from a company's latest SEC 10-K, Yahoo Finance market data, and Groq. The Python backend serves a FastAPI WebSocket and read-only run-history endpoints; the Next.js frontend shows progress, source excerpts, and past runs.

## Local setup

Requires Python 3.13+, `uv`, Node.js, and a Groq API key for live analyses. Install dependencies:

```sh
uv sync --group dev
cd frontend && npm ci
```

Set these environment variables (or place them in a local `.env`, which is ignored):

```sh
GROQ_API_KEY=your-key
SEC_USER_AGENT="DueDiligenceAI your-contact@example.com"
# Optional; defaults to sqlite+aiosqlite:///./due_diligence.db
DATABASE_URL="sqlite+aiosqlite:///./due_diligence.db"
```

The SEC requires a real identifying User-Agent with a contact email; replace the example with your own contact. A local `sentence-transformers/all-MiniLM-L6-v2` model is downloaded on the first analysis if it is not already cached. There is no hosted embedding API or separate vector database. Start the backend from the project root with `uv run uvicorn server:api --reload` and the frontend with `cd frontend && npm run dev`; open `http://localhost:3000`. Live analysis makes external SEC, Yahoo Finance, and Groq requests. Running the offline tests or benchmark does not.

The current Groq model is `openai/gpt-oss-120b` (approved after the previous `llama-3.3-70b-versatile` model returned HTTP 404 for this account).

## Pipeline and API

1. `data.SECClient` resolves the ticker to a CIK and checks SEC submissions for the latest 10-K accession. It selects the primary HTML filing, removes hidden/iXBRL markup, and extracts body Items 1, 1A, 7, and 8. Section boundaries skip repeated running page headers and table-of-contents entries; missing boundaries fail explicitly rather than silently swallowing neighboring sections.
2. `retrieval.make_passages` preserves bounded excerpts and accession/item/index IDs. The local embedding model generates normalized passage vectors cached in SQLite by accession, model, and chunker version. For each item, focused questions retrieve their top two passages by cosine similarity; the results are deduplicated and capped at six passages before a single Groq extraction call. Only passages from that item enter the prompt. This is passage-level retrieval-augmented generation, not whole-document prompting or a semantic search across all filings.
3. The LangGraph pipeline loads fundamentals and the filing, fans out into section summaries, market-risk calculation, and insider activity, then joins them for Groq synthesis. Filing claims must reference IDs in the selected evidence; unknown IDs fail validation. The final brief contains SEC links and a separate verbatim excerpt for each cited passage. The URL points to the filing, not an exact HTML anchor. The model's factual correctness still needs human review; valid citation IDs alone do not prove every assertion.
4. `storage.Store` uses async SQLAlchemy and SQLite (`PRAGMA user_version` migration) for filing/passage caches and runs. The latest SEC accession is refreshed on each new run before reusing cached content. A parser-version change reparses cached source text and invalidates its old passages/embeddings; model and chunker versions key the embedding cache. Existing version-1 SQLite files migrate in place. The frontend reads history from the backend, not browser localStorage. It shows up to 20 recent runs; the database retains the records.

Connect to `ws://localhost:8000/test`, then send JSON `{"ticker":"AAPL"}`. The server emits per-node completion events `{"type":"stage","stage":"fundamentals|summary|risk|info|synthesis","status":"completed"}`, followed by `{"type":"final","run":{...}}` or `{"type":"error","run":{...},"message":"..."}`. Errors before run creation have `"run":null`. A run includes `id`, `ticker`, `accession`, `status`, `startedAt`, `finishedAt`, `finalBrief`, `citations` (`id`, `item`, `url`, `excerpt`), and `error`. The final brief arrives as a whole: **individual model tokens are not streamed**. `GET /runs` lists the 20 newest runs; `GET /runs/{id}` returns full run detail (404 if absent). The only accepted WebSocket ticker syntax is 1–10 uppercase alphanumeric/dot/hyphen characters after normalization.

## Offline verification and performance

```sh
uv run pytest -q
uv run python -m scripts.legacy_baseline > /tmp/diligence-legacy.json
uv run python -m scripts.benchmark --label local-mock > /tmp/diligence-local-mock.json
uv run python -m scripts.evaluate_retrieval
cd frontend && npm run build
```

Tests use synthetic SEC-shaped fixtures and fake SEC/Yahoo/Groq/model responses, covering SEC 429 handling, heading boundaries, citation validation, graph joins, two simultaneously active WebSockets, disconnects, SQLite restarts, and cache invalidation. All 22 backend tests passed without credentials. Both benchmark scripts use the same invented four-item filing and Yahoo inputs, a controlled 2 ms delay on each of five fake model calls per run, and 21 completed runs (zero failed timed runs); their JSON includes raw per-stage timings and environment. The historical graph has no cache and returns mock plain text; the new graph has a cold first run followed by 20 warm SQLite-cache runs and mock citation-bearing JSON. They therefore **cannot establish a production speedup**. The local MiniLM retrieval check ranked the expected passage first for 4/4 hand-labeled synthetic questions (recall@1); that tiny invented dataset is not evidence of accuracy on real 10-Ks. No 30-minute manual baseline or real-world speedup has been measured.

| Offline mock workflow (Python 3.13.5, macOS 27.0.1 arm64) | Median wall time | p95 wall time |
| --- | ---: | ---: |
| Historical, 21 uncached runs | 13.864 ms | 14.538 ms |
| Current, 1 cold + 20 warm runs | 8.979 ms | 9.287 ms |

These are one local run of each harness, not comparable user-facing throughput figures. The historical summary-stage median was 9.932 ms and the current summary-stage median was 4.620 ms; all stages and individual runs are in the harness JSON. The actual Next.js dashboard was opened against the fake-backed FastAPI server: it rendered a completed ACME brief with an SEC excerpt, and the completed run remained selectable from history after reload. No live Groq response was checked.

In a separate SEC-only smoke check, the 2026 Apple 10-K yielded four sections and 93 indexed passages. The focused Item 8 queries selected five distinct passages containing net-income, net-sales, and balance-sheet text; the selection was inspected with a fake Groq client, not a generated live brief. This demonstrates context coverage on one filing, not citation accuracy or model answer quality.

A separate non-paid Microsoft 10-K check exposed truncated sections caused by repeated running page headers. The parser now keeps each item through its next distinct SEC item, and the cached filing is reparsed on upgrade; a synthetic multi-page fixture covers the regression. These checks establish section coverage, not financial answer accuracy.

### Live retrieval evaluation

Set a real `SEC_USER_AGENT` as in Local setup. Run `uv run python -m scripts.evaluate_retrieval_live` for pinned 2026 Apple/Microsoft filings; add `--dataset holdout` for Alphabet/Berkshire or `--dataset fresh` for Meta/Amazon. Each set has eight manually labeled source-text anchors, one per filing item. The script fails when an accession or anchor changes. It compares the frozen original MiniLM query/selection strategy, an **offline-only** MiniLM + BM25 experiment using the original queries, and the live diversified MiniLM questions with one first-choice passage per question before filling the remaining slots. All contexts remain capped at six passages per item; the four Groq section calls are unchanged.

| Dataset | Strategy | Source passage in top 4 | Source passage in selected context |
| --- | --- | ---: | ---: |
| Apple + Microsoft (development) | Frozen original | 6 / 8 | 3 / 8 |
| Apple + Microsoft (development) | Experimental hybrid | 4 / 8 | 3 / 8 |
| Apple + Microsoft (development) | Live diversified | 6 / 8 | 5 / 8 |
| Alphabet + Berkshire (initial holdout) | Frozen original | 7 / 8 | 3 / 8 |
| Alphabet + Berkshire (initial holdout) | Experimental hybrid | 6 / 8 | 2 / 8 |
| Meta + Amazon (fresh holdout) | Frozen original | 6 / 8 | 4 / 8 |
| Meta + Amazon (fresh holdout) | Experimental hybrid | 6 / 8 | 4 / 8 |
| Meta + Amazon (fresh holdout) | Live diversified | 6 / 8 | 5 / 8 |

Hybrid ranking regressed on the initial two sets and was removed from the live agent; its ranker remains dev-only to reproduce the comparison. The initial holdout informed the decision to try question diversification, so **only Meta/Amazon is independent of that decision**. Diversification improved source-passage coverage by 2/8 on development filings and 1/8 on the fresh holdout; it increases local query embedding work, not Groq call count or passage cap. These tiny, manually selected samples are not representative accuracy estimates, financial fact-checks, or generated-answer scores. No Groq calls or paid APIs were used in these measurements.

### Live Groq smoke result

On 2026-11-16, the configured Groq key successfully listed models, and `openai/gpt-oss-120b` returned citation-bearing JSON for a minimal request. A full AAPL WebSocket analysis **did not complete**: one section response failed strict citation-ID validation; its run was correctly persisted as failed. A diagnostic run then returned three section responses with valid citation IDs, but Groq rejected another section request with HTTP 429 at the account's 8,000 tokens/minute limit (6,835 already used; 4,367 requested). There is no verified live final brief or live claim-accuracy result. Do not weaken citation validation to mask this failure.

## Limits

This is a single-instance, single-user local app with no authentication or per-user isolation. Do not deploy it publicly without making an access-control decision. SQLite and in-process cosine ranking fit one filing at a time, not a multi-instance search service. SEC HTML and model outputs vary; missing filing headings, insufficient evidence, rate limits, and upstream failures produce a failed run rather than a fabricated answer. The source-link validator checks passage IDs and does not replace financial fact-checking. First-run model download and live Groq calls require connectivity; the automated checks do not use paid APIs.
