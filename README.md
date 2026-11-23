# Due-diligence AI

A single-user local application that builds a source-linked due-diligence brief from a company's latest SEC 10-K, Yahoo Finance market data, and OpenRouter. The Python backend serves a FastAPI WebSocket and read-only run-history endpoints; the Next.js frontend shows progress, source excerpts, and past runs.

## Local setup

Requires Python 3.13+, `uv`, Node.js, and an OpenRouter API key for live analyses. Install dependencies:

```sh
uv sync --group dev
cd frontend && npm ci
```

Copy `.env.example` to the ignored `.env` manually only if you do not already have a `.env`; otherwise add the required settings to your existing file without overwriting its secrets:

```sh
OPENROUTER_API_KEY=your-openrouter-key
SEC_USER_AGENT="DueDiligenceAI your-contact@example.com"
# Optional; defaults to sqlite+aiosqlite:///./due_diligence.db
DATABASE_URL="sqlite+aiosqlite:///./due_diligence.db"
```

Use an OpenRouter key explicitly: `GROQ_API_KEY` and `OPENAI_API_KEY` are not substitutes or fallbacks. The SEC requires a real identifying User-Agent with a contact email; replace the example with your own contact. A local `sentence-transformers/all-MiniLM-L6-v2` model is downloaded on the first analysis if it is not already cached. There is no hosted embedding API or separate vector database. Start the backend from the project root with `uv run uvicorn server:api --reload` and the frontend with `cd frontend && npm run dev`; open `http://localhost:3000`. Live analysis makes external SEC, Yahoo Finance, and OpenRouter requests. Running the offline tests or benchmark does not.

The OpenRouter model is `openai/gpt-oss-120b`. Requests use strict JSON Schema with a per-request enum of permitted citation IDs; local citation validation still rejects unsupported IDs. Provider routing requires parameter support (`require_parameters=true`), disallows fallback providers (`allow_fallbacks=false`), and denies provider data collection (`data_collection=deny`). No implicit provider or API-key switching is configured. One fully live AAPL analysis completed after Yahoo Finance was configured to use Firefox impersonation; see the live smoke results below.

## Pipeline and API

1. `data.SECClient` resolves the ticker to a CIK and checks SEC submissions for the latest 10-K accession. It selects the primary HTML filing, removes hidden/iXBRL markup, and extracts body Items 1, 1A, 7, and 8. Section boundaries skip repeated running page headers and table-of-contents entries; missing boundaries fail explicitly rather than silently swallowing neighboring sections.
2. `retrieval.make_passages` preserves bounded excerpts and accession/item/index IDs. The local embedding model generates normalized passage vectors cached in SQLite by accession, model, and chunker version. For each item, focused questions retrieve their top two passages by cosine similarity; the results are deduplicated and capped at six passages before a single OpenRouter extraction call. Only passages from that item enter the prompt. This is passage-level retrieval-augmented generation, not whole-document prompting or a semantic search across all filings.
3. The LangGraph pipeline loads fundamentals and the filing, fans out into section summaries, market-risk calculation, and insider activity, then joins them for OpenRouter synthesis. Filing claims must reference IDs in the selected evidence; unknown IDs fail validation. The final brief contains SEC links and a separate verbatim excerpt for each cited passage. The URL points to the filing, not an exact HTML anchor. The model's factual correctness still needs human review; valid citation IDs alone do not prove every assertion.
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

`uv run pytest -q` passed 25 credential-free backend tests (one dependency deprecation warning); `cd frontend && npm run build` passed compilation and TypeScript. The two offline benchmark scripts each timed 21 completed runs against the same invented SEC/Yahoo inputs, with five fake model calls per run delayed by 2 ms each; neither benchmark calls an external API. The historical graph has no cache and returns mock plain text. The current graph returns mock citation-bearing JSON and uses SQLite: one cold run, then 20 warm runs. Both scripts emit raw per-stage timings and environment details.

| Offline mock workflow (Python 3.13.5, macOS 27.2.1 arm64) | Cache state | Median run | p95 run |
| --- | --- | ---: | ---: |
| Historical graph, 21 runs | No cache | 14.301 ms | 16.262 ms |
| Current graph, 21 runs | 1 cold + 20 warm SQLite runs | 9.048 ms | 9.531 ms |

The median is the middle of 21 runs; p95 is the 20th-fastest run. **Do not interpret the lower current mock times as a production speedup:** the old graph is always uncached, the new graph is mostly warm, the generated response formats differ, and real network/model delays are replaced by fixed fake delays. Actual OpenRouter request timings, usage, and cost are reported separately below and cannot be compared to these simulated milliseconds. No real-world speedup or roughly 30-minute manual baseline has been measured. An earlier four-question synthetic retrieval check found the expected passage first in 4/4 invented cases; that is not real-filing accuracy. The Next.js dashboard was previously opened with a fake-backed FastAPI server: it displayed a cited brief and restored it from backend history after reload.

In a separate SEC-only smoke check, the 2026 Apple 10-K yielded four sections and 93 indexed passages. The focused Item 8 queries selected five distinct passages containing net-income, net-sales, and balance-sheet text; the selection was inspected with a fake model client, not a generated live brief. This demonstrates context coverage on one filing, not citation accuracy or model answer quality.

A separate non-paid Microsoft 10-K check exposed truncated sections caused by repeated running page headers. The parser now keeps each item through its next distinct SEC item, and the cached filing is reparsed on upgrade; a synthetic multi-page fixture covers the regression. These checks establish section coverage, not financial answer accuracy.

### Live retrieval evaluation

Set a real `SEC_USER_AGENT` as in Local setup. Run `uv run python -m scripts.evaluate_retrieval_live` for pinned 2026 Apple/Microsoft filings; add `--dataset holdout` for Alphabet/Berkshire or `--dataset fresh` for Meta/Amazon. Each set has eight manually labeled source-text anchors, one per filing item. The script fails when an accession or anchor changes. It compares the frozen original MiniLM query/selection strategy, an **offline-only** MiniLM + BM25 experiment using the original queries, and the live diversified MiniLM questions with one first-choice passage per question before filling the remaining slots. All contexts remain capped at six passages per item; the four OpenRouter section calls are unchanged.

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

Hybrid ranking regressed on the initial two sets and was removed from the live agent; its ranker remains dev-only to reproduce the comparison. The initial holdout informed the decision to try question diversification, so **only Meta/Amazon is independent of that decision**. Diversification improved source-passage coverage by 2/8 on development filings and 1/8 on the fresh holdout; it increases local query embedding work, not OpenRouter call count or passage cap. These tiny, manually selected samples are not representative accuracy estimates, financial fact-checks, or generated-answer scores. No model calls or paid APIs were used in these measurements.
The latest evaluation rechecked all six pinned SEC filings and reproduced the counts above.

### Historical Groq live smoke result

On 2026-11-16, the then-configured Groq key successfully listed models, and `openai/gpt-oss-120b` returned citation-bearing JSON for a minimal request. A full AAPL WebSocket analysis **did not complete**: one section response failed strict citation-ID validation; its run was correctly persisted as failed. A diagnostic run then returned three section responses with valid citation IDs, but Groq rejected another section request with HTTP 429 at the account's 8,000 tokens/minute limit (6,835 already used; 4,367 requested). These were Groq-era results, not an OpenRouter live test; no live Groq final brief or generated-claim accuracy was verified. Do not weaken citation validation to mask this failure.

### OpenRouter live smoke result

A small live request to `openai/gpt-oss-120b` returned a valid strict-schema claim with the supplied citation ID; OpenRouter reported 172 total tokens and a cost of $0.00001188 for that request. Before the Firefox change, a fully live AAPL WebSocket run failed before model generation because Yahoo Finance's default Chrome-impersonating `curl_cffi` session raised a TLS `invalid library` error. Its failed run was available through `GET /runs/{id}`.

A separate diagnostic AAPL WebSocket run replaced only Yahoo fundamentals and insider transactions with deterministic fixtures. Real SEC filing retrieval/parsing (accession `0000320193-26-000026`), local embedding and passage selection, OpenRouter section extraction and synthesis, and SQLite run persistence remained active. It completed in 64.5 seconds with 15 citation records and a 6,944-character brief; `GET /runs/{id}` returned the completed run while the temporary test database existed. This verifies that isolated path, not live Yahoo integration, model factual accuracy, or production latency. No Yahoo code was changed.

After switching all Yahoo Finance calls to a shared Firefox-impersonating session, a fully live AAPL WebSocket run completed using Yahoo prices, fundamentals, and insider transactions, the SEC filing, local passage retrieval, and OpenRouter generation. It took 76.8 seconds in this single local run; the completed brief had 19 citation records and 8,148 characters. `GET /runs/{id}` returned the completed run while the temporary test database existed. This is an integration smoke result, not a model factual-accuracy assessment or a representative latency benchmark.

### Live OpenRouter evaluations

Two analyses (AAPL and META) ran sequentially with a temporary SQLite database; a separate MSFT run instrumented the five actual OpenRouter requests. All three used real Yahoo Finance, SEC filings, local retrieval, and OpenRouter. Each completed through the WebSocket, and `GET /runs/{id}` returned its completed record before the temporary database was removed. Times are wall-clock measurements of individual runs, **not** a latency distribution or a comparison with the offline mock benchmark.

| Ticker | WebSocket duration | Claims with filing citations / all claims | Source links resolving to saved excerpts | Run |
| --- | ---: | ---: | ---: | --- |
| AAPL | 37.889 s | 25 / 36 | 30 / 30 | Completed, HTTP 200 on history lookup |
| META | 28.063 s | 18 / 29 | 20 / 20 | Completed, HTTP 200 on history lookup |
| MSFT | 41.961 s | 24 / 32 | 26 / 26 | Completed, HTTP 200 on history lookup |

#### Actual OpenRouter requests in the MSFT run

| Request | Latency | Input tokens | Output tokens | OpenRouter-reported cost |
| --- | ---: | ---: | ---: | ---: |
| Item 1 extraction | 4.786 s | 2,699 | 349 | $0.00013236 |
| Item 1A extraction | 4.990 s | 2,055 | 352 | $0.00011373 |
| Item 7 extraction | 6.113 s | 2,787 | 436 | $0.00014805 |
| Item 8 extraction | 5.991 s | 2,413 | 458 | $0.00014037 |
| Final synthesis | 19.654 s | 3,201 | 2,004 | $0.00039591 |

All five real requests finished successfully with strict JSON Schema. OpenRouter reported **13,155 input tokens, 3,599 output tokens, and $0.00093042 total** for this MSFT run. The four section requests overlap under bounded concurrency, so adding their latencies does not yield the 41.961-second end-to-end duration, which also includes SEC, Yahoo, local embeddings, and storage. These are single-run observations, not a provider latency or cost distribution.

Source links resolve to allowed passage IDs, but **the generated claims are not reliably grounded**. In a manual spot-check, an Apple claim called $67,656 million the company's operating income, although its cited Item 8 table lists that as one segment and $123,216 million as the table total. Another Apple claim called $12,815 million lease liabilities; its cited table calls that right-of-use *assets* and lists $14,368 million of lease liabilities. A Meta claim described accelerating Reality Labs hardware adoption, but its cited excerpt describes sales efforts without an adoption trend. These examples demonstrate unsupported or misread claims, not an exhaustive error count or an accuracy percentage. Review any generated financial statement against the filing before use; valid citation IDs alone do not make it correct.

The MSFT run also cited an Item 7 passage about tax rates for a 27% Microsoft Cloud revenue-growth claim; that passage contains no cloud revenue figure. Three completed runs demonstrate API integration, not generated-answer accuracy.

## Limits

This is a single-instance, single-user local app with no authentication or per-user isolation. Do not deploy it publicly without making an access-control decision. SQLite and in-process cosine ranking fit one filing at a time, not a multi-instance search service. SEC HTML and model outputs vary; missing filing headings, insufficient evidence, rate limits, and upstream failures produce a failed run rather than a fabricated answer. The source-link validator checks passage IDs and does not replace financial fact-checking. First-run model download and live OpenRouter calls require connectivity; the automated checks do not use paid APIs.
