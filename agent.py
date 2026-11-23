"""LangGraph analysis pipeline; dependencies are supplied per request, not at import."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from statistics import stdev
from typing import Any, TypedDict

import yfinance as fin
from openai import AsyncOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

import data
from retrieval import LocalEmbedder, Passage, make_passages, MODEL_VERSION, CHUNKER_VERSION
from storage import Store


SECTIONS = {
    "Item 1": (
        "What are the company's products, services, and revenue segments?",
        "Which geographic markets and customer groups does the company serve?",
        "What customer support, warranty, or subscription services are offered?",
        "What cloud platforms or other technology services are offered?",
    ),
    "Item 1A": (
        "What risks could materially affect sales or financial performance?",
        "What supply, regulatory, or competitive risks does the filing describe?",
        "What privacy, data protection, or cybersecurity risks could affect customers?",
    ),
    "Item 7": (
        "How did net sales and revenue change from the prior fiscal year?",
        "What changed in gross margin and operating costs?",
        "How did operating income or segment operating earnings change?",
        "What outlook or forward guidance does management explicitly provide?",
    ),
    "Item 8": (
        "What are consolidated net sales and net income for the fiscal year?",
        "What do the income statement and gross margin figures report?",
        "What cash, assets, liabilities, or debt appear on the balance sheet?",
    ),
}
OPENROUTER_MODEL = "openai/gpt-oss-120b"
OPENROUTER_ROUTE = {
    "provider": {
        "require_parameters": True,
        "allow_fallbacks": False,
        "data_collection": "deny",
    },
}


def _claim_schema(allowed: set[str]) -> dict[str, Any]:
    citation = (
        {"type": "string", "enum": sorted(allowed)}
        if allowed else {"type": "string"}
    )
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "citations": {"type": "array", "items": citation},
        },
        "required": ["text", "citations"],
        "additionalProperties": False,
    }


@dataclass
class AnalysisContext:
    sec: data.SECClient
    store: Store
    embedder: LocalEmbedder
    llm: AsyncOpenAI | None
    executor: ThreadPoolExecutor
    model_limit: asyncio.Semaphore
    index_locks: tuple[asyncio.Lock, ...]


class AgentState(TypedDict, total=False):
    ticker: str
    fundamentals: dict[str, Any]
    filing_ref: data.FilingRef
    passages: list[Passage]
    risk_info: dict[str, float]
    info_collected: dict[str, Any]
    summary_claims: list[dict[str, Any]]
    ten_k_summary: str
    final_output: str
    citations: list[dict[str, str]]


def _context(config: RunnableConfig) -> AnalysisContext:
    return config["configurable"]["context"]


async def _finance_call(ctx: AnalysisContext, function, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(ctx.executor, function, *args)


async def fundamentals_agent(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _context(config)
    ticker = state["ticker"]
    vitals_task = asyncio.create_task(_finance_call(ctx, data.getVitals, ticker))
    try:
        ref = await ctx.sec.resolve_filing(ticker)
        cached = await ctx.store.get_filing(ref.accession)
        if cached is None:
            filing = await ctx.sec.fetch_filing(ref)
            await ctx.store.save_filing(
                ref, filing.text, filing.sections, data.PARSER_VERSION
            )
        lock_key = hash((ref.accession, MODEL_VERSION, CHUNKER_VERSION))
        async with ctx.index_locks[lock_key % len(ctx.index_locks)]:
            # A second run may have populated or upgraded this accession while
            # the first was fetching it. Re-read before invalidating passages.
            cached = await ctx.store.get_filing(ref.accession)
            if cached["parser_version"] != data.PARSER_VERSION:
                sections = data.slice_10k_specific_items(cached["text"])
                await ctx.store.update_sections(ref.accession, sections, data.PARSER_VERSION)
            else:
                sections = cached["sections"]

            passages = await ctx.store.get_passages(ref.accession, MODEL_VERSION, CHUNKER_VERSION)
            if not passages:
                passages = make_passages(ref.accession, sections)
                vectors = await ctx.embedder.embed([passage.text for passage in passages])
                if len(vectors) != len(passages):
                    raise ValueError("Embedding count differs from passage count")
                passages = [
                    replace(passage, embedding=vector)
                    for passage, vector in zip(passages, vectors)
                ]
                await ctx.store.save_passages(ref.accession, passages, MODEL_VERSION, CHUNKER_VERSION)
        vitals = await vitals_task
        return {"fundamentals": vitals, "filing_ref": ref, "passages": passages}
    finally:
        if not vitals_task.done():
            vitals_task.cancel()
            await asyncio.gather(vitals_task, return_exceptions=True)


def _parse_claims(raw: str | None, allowed: set[str]) -> list[dict[str, Any]]:
    if not raw:
        raise ValueError("Empty model response")
    result = json.loads(raw)
    claims = result.get("claims")
    if not isinstance(claims, list):
        raise ValueError("Model response lacks claims")
    validated = []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            raise ValueError("Invalid model claim")
        text = claim["text"].strip()
        ids = claim.get("citations")
        if not text or not isinstance(ids, list) or not ids or any(
            not isinstance(identifier, str) or identifier not in allowed for identifier in ids
        ):
            raise ValueError("Model claim has missing or invalid source citations")
        validated.append({"text": text, "citations": list(dict.fromkeys(ids))})
    return validated


async def select_passages(
    embedder: LocalEmbedder, passages: list[Passage], item: str,
    questions: tuple[str, ...],
) -> dict[str, Passage]:
    """Reserve one evidence slot per question before filling remaining slots."""
    ranked = [await embedder.search(question, passages, item, k=2) for question in questions]
    selected: dict[str, Passage] = {}
    for rank in range(2):
        for results in ranked:
            if len(selected) == 6:
                return selected
            if len(results) > rank:
                passage = results[rank]
                selected.setdefault(passage.id, passage)
    return selected


async def summary_agent(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _context(config)

    async def summarize(item: str, questions: tuple[str, ...]) -> tuple[str, list[dict[str, Any]]]:
        selected = await select_passages(ctx.embedder, state["passages"], item, questions)
        if not selected:
            return item, []
        allowed = set(selected)
        evidence = "\n\n".join(f"[{passage.id}] {passage.text}" for passage in selected.values())
        async with ctx.model_limit:
            response = await ctx.llm.chat.completions.create(
                model=OPENROUTER_MODEL,
                timeout=60,
                max_tokens=2048,
                reasoning_effort="low",
                extra_body=OPENROUTER_ROUTE,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "filing_claims",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "claims": {"type": "array", "items": _claim_schema(allowed)},
                            },
                            "required": ["claims"],
                            "additionalProperties": False,
                        },
                    },
                },
                messages=[
                    {"role": "system", "content": (
                        "You extract facts ONLY from the supplied 10-K passages. Return a JSON object "
                        "with key claims: an array of {text, citations}. Each factual claim needs at least "
                        "one exact passage ID in citations. Never invent an ID or a fact. If there is no "
                        "answer in the supplied passages, return {\"claims\":[]}. Maximum five concise claims."
                    )},
                    {"role": "user", "content": (
                        f"Item: {item}\nQuestions:\n- " + "\n- ".join(questions)
                        + f"\nPassages:\n{evidence}"
                    )},
                ],
            )
        return item, _parse_claims(response.choices[0].message.content, allowed)

    results = await asyncio.gather(*(summarize(item, queries) for item, queries in SECTIONS.items()))
    claims = [dict(claim, item=item) for item, section in results for claim in section]
    compiled = "\n".join(
        f"{claim['item']}: {claim['text']} " + " ".join(f"[[{id_}]]" for id_ in claim["citations"])
        for claim in claims
    ) or "Insufficient filing evidence in selected passages."
    return {"summary_claims": claims, "ten_k_summary": compiled}


async def risk_agent(state: AgentState) -> dict:
    fundam = state["fundamentals"]
    prices = fundam["priceHist"]
    returns = fundam["dailyReturns"]
    if len(prices) < 2 or len(returns) < 2:
        raise ValueError("Not enough price observations to calculate risk")
    maximum = prices[0]
    drawdown = 0.0
    for price in prices:
        maximum = max(maximum, price)
        drawdown = min(drawdown, (price - maximum) / maximum)
    return {"risk_info": {
        "volatility": stdev(returns),
        "beta": fundam["beta"],
        "max_drawdown": drawdown,
    }}


def _insider_transactions(ticker: str) -> list[dict]:
    table = fin.Ticker(ticker, session=data.yahoo_session()).insider_transactions
    if table is None or table.empty:
        return []
    return table.astype(str).head(20).to_dict(orient="records")


async def info_agent(state: AgentState, config: RunnableConfig) -> dict:
    transactions = await _finance_call(_context(config), _insider_transactions, state["ticker"])
    return {"info_collected": {"insider_trans": transactions}}


def _parse_brief(raw: str | None, evidence: dict[str, Passage], url: str) -> tuple[str, list[dict[str, str]]]:
    if not raw:
        raise ValueError("Empty synthesis response")
    result = json.loads(raw)
    sections = result.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("Synthesis response lacks sections")
    lines: list[str] = []
    used: dict[str, Passage] = {}
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("heading"), str):
            raise ValueError("Invalid synthesis heading")
        claims = section.get("claims")
        if not isinstance(claims, list):
            raise ValueError("Invalid synthesis claims")
        lines.append(f"## {section['heading'].strip()}")
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                raise ValueError("Invalid synthesis claim")
            text = claim["text"].strip()
            ids = claim.get("citations")
            if not text or not isinstance(ids, list) or any(
                not isinstance(identifier, str) or identifier not in evidence for identifier in ids
            ):
                raise ValueError("Synthesis refers to missing source evidence")
            for identifier in ids:
                used[identifier] = evidence[identifier]
            links = " ".join(f"[source {identifier}]({url})" for identifier in dict.fromkeys(ids))
            lines.append(f"- {text}" + (f" {links}" if links else ""))
        lines.append("")
    if evidence and not used:
        raise ValueError("Synthesis omitted available filing citations")
    citations = [{"id": id_, "item": passage.item, "url": url, "excerpt": passage.text}
                 for id_, passage in used.items()]
    return "\n".join(lines).strip(), citations


async def synth_agent(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _context(config)
    fundam = state["fundamentals"]
    ref = state["filing_ref"]
    claims = state["summary_claims"]
    used_ids = {id_ for claim in claims for id_ in claim["citations"]}
    evidence = {passage.id: passage for passage in state["passages"] if passage.id in used_ids}
    citation_schema = _claim_schema(set(evidence))
    prompt = {
        "ticker": state["ticker"],
        "sector": fundam["sector"],
        "peRatio": fundam["peRatio"],
        "marketCap": fundam["marketCap"],
        "risk": state["risk_info"],
        "insider_transactions": state["info_collected"]["insider_trans"],
        "filing_claims": claims,
    }
    async with ctx.model_limit:
        response = await ctx.llm.chat.completions.create(
            model=OPENROUTER_MODEL,
            timeout=60,
            max_tokens=3072,
            reasoning_effort="low",
            extra_body=OPENROUTER_ROUTE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "due_diligence_brief",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "heading": {"type": "string"},
                                        "claims": {
                                            "type": "array",
                                            "items": citation_schema,
                                        },
                                    },
                                    "required": ["heading", "claims"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["sections"],
                        "additionalProperties": False,
                    },
                },
            },
            messages=[
                {"role": "system", "content": (
                    "Write a concise financial due-diligence brief as JSON with key sections: "
                    "an array of {heading, claims:[{text,citations}]}. Include Company Snapshot, "
                    "Financial Analysis, Risk Assessment, Insider Activity, Bull Case, Bear Case, "
                    "and Recommendation (Buy/Hold/Sell with rationale). For every fact taken from "
                    "filing_claims, include its existing passage IDs in citations; market metrics and "
                    "insider facts may have empty citations. Do not add facts not in the input, invent "
                    "citations, or infer unsupported details; say when evidence is insufficient. "
                    "Keep numerical units correct: volatility is percent daily returns, drawdown a fraction."
                )},
                {"role": "user", "content": json.dumps(prompt, default=str)},
            ],
        )
    brief, citations = _parse_brief(response.choices[0].message.content, evidence, ref.url)
    return {"final_output": brief, "citations": citations}


graph = StateGraph(AgentState)
graph.add_node("fundamentals_agent", fundamentals_agent)
graph.add_node("summary_agent", summary_agent)
graph.add_node("risk_agent", risk_agent)
graph.add_node("info_agent", info_agent)
graph.add_node("synth_agent", synth_agent)
graph.add_edge(START, "fundamentals_agent")
for node in ("summary_agent", "risk_agent", "info_agent"):
    graph.add_edge("fundamentals_agent", node)
    graph.add_edge(node, "synth_agent")
graph.add_edge("synth_agent", END)
app = graph.compile()
