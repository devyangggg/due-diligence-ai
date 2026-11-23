"""Measure evidence coverage on pinned SEC filings without Groq calls.

Requires SEC_USER_AGENT with a real contact email. Each manually labeled anchor
identifies a source passage, not a generated answer. The fresh META/AMZN holdout
was labeled after freezing the diversified live selection algorithm.
Run: SEC_USER_AGENT='App contact@example.com' uv run python -m scripts.evaluate_retrieval_live
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import replace

import httpx
from dotenv import load_dotenv

from agent import SECTIONS, select_passages
from data import SECClient
from retrieval import LocalEmbedder, make_passages
from scripts.hybrid_experiment import HybridRetriever

BASELINE_SECTIONS = {
    "Item 1": (
        "What are the company's products, services, and revenue segments?",
        "Which geographic markets and customer groups does the company serve?",
    ),
    "Item 1A": (
        "What risks could materially affect sales or financial performance?",
        "What supply, regulatory, or competitive risks does the filing describe?",
    ),
    "Item 7": (
        "How did net sales and revenue change from the prior fiscal year?",
        "What changed in gross margin, costs, or operating income?",
        "What outlook or forward guidance does management explicitly provide?",
    ),
    "Item 8": (
        "What are consolidated net sales and net income for the fiscal year?",
        "What do the income statement and gross margin figures report?",
        "What cash, assets, liabilities, or debt appear on the balance sheet?",
    ),
}


PINNED = {
    "AAPL": "0000320193-26-000026",
    "MSFT": "0001193125-26-323660",
}
# Each label identifies the source passage, not the answer expected from a model.
CASES = [
    ("AAPL", "Item 1", "What support services do customers receive?", r"AppleCare® brand"),
    ("AAPL", "Item 1A", "How could handling personal data damage customer trust?",
     r"privacy and data security concerns that could erode customer trust"),
    ("AAPL", "Item 7", "How much did total net sales grow in 2026?",
     r"Total net sales\s*\$\s*478,834"),
    ("AAPL", "Item 8", "What was net income for fiscal 2026?",
     r"Net income\s*\$\s*132,175"),
    ("MSFT", "Item 1", "Which cloud platform does Microsoft offer?",
     r"Azure is a comprehensive set of cloud services"),
    ("MSFT", "Item 1A", "How could a security breach affect Microsoft's customers?",
     r"The security of our products and services is important"),
    ("MSFT", "Item 7", "How did operating income change in 2026?",
     r"Operating income increased\s*\$26\.7 billion or 21%"),
    ("MSFT", "Item 8", "How much net income was reported for fiscal 2026?",
     r"Net income\s*\$\s*133,749"),
]

HOLDOUT_PINNED = {
    "GOOGL": "0001652044-26-000018",
    "BRK-B": "0001193125-26-083899",
}
HOLDOUT_CASES = [
    ("GOOGL", "Item 1", "What services does YouTube provide?",
     r"YouTube provides people with entertainment, information"),
    ("GOOGL", "Item 1A", "How could advertising privacy changes affect business?",
     r"data privacy practices have in the past"),
    ("GOOGL", "Item 7", "How did total revenues change in 2025?",
     r"Total revenues\s*\$\s*350,018\s*\$\s*402,836"),
    ("GOOGL", "Item 8", "What was net income in 2025?",
     r"Net income\s*\$\s*73,795\s*\$\s*100,118\s*\$\s*132,170"),
    ("BRK-B", "Item 1", "What are Berkshire's principal businesses?",
     r"insurance businesses, conducted on both a primary basis and a reinsurance basis"),
    ("BRK-B", "Item 1A", "How could catastrophic events affect Berkshire insurance losses?",
     r"significant loss from a single catastrophe event"),
    ("BRK-B", "Item 7", "What earnings did the insurance underwriting segment generate?",
     r"Insurance underwriting generated after-tax earnings of \$7\.3 billion"),
    ("BRK-B", "Item 8", "What were consolidated net earnings in 2025?",
     r"Net earnings\s+67,260\s+89,561\s+97,147"),
]

FRESH_PINNED = {
    "META": "0001628280-26-003942",
    "AMZN": "0001018724-26-000004",
}
FRESH_CASES = [
    ("META", "Item 1", "What messaging product does Meta offer?",
     r"WhatsApp is a simple, reliable, and secure messaging application"),
    ("META", "Item 1A", "What security risks threaten Meta users' data?",
     r"security breaches, improper access to or disclosure of our data or user data"),
    ("META", "Item 7", "How did income from operations change in 2025?",
     r"Income from operations for 2025 was \$83\.28 billion"),
    ("META", "Item 8", "What was consolidated net income in 2025?",
     r"Net income\s*\$\s*60,458\s*\$\s*62,360\s*\$\s*39,098"),
    ("AMZN", "Item 1", "What benefits does Amazon Prime membership provide?",
     r"Amazon Prime, a membership program that includes fast, free shipping"),
    ("AMZN", "Item 1A", "How could unauthorized access compromise Amazon customer data?",
     r"prevent, detect, or mitigate data loss, theft, misuse, unauthorized access"),
    ("AMZN", "Item 7", "How did operating income change in 2025?",
     r"Operating income was \$68\.6 billion and \$80\.0 billion"),
    ("AMZN", "Item 8", "What were Amazon's total net sales in 2025?",
     r"Total net sales\s+574,785\s+637,959\s+716,924"),
]


async def context_ids(search, item):
    selected = {}
    for focused_query in BASELINE_SECTIONS[item]:
        for passage in await search(focused_query, item, k=2):
            if len(selected) >= 6:
                break
            selected.setdefault(passage.id, passage)
    return list(selected)


async def evaluate(dataset="development"):
    datasets = {
        "development": (PINNED, CASES),
        "holdout": (HOLDOUT_PINNED, HOLDOUT_CASES),
        "fresh": (FRESH_PINNED, FRESH_CASES),
    }
    pinned_filings, cases = datasets[dataset]
    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise SystemExit("Set SEC_USER_AGENT to an identifying app name and contact email")
    model = LocalEmbedder()
    rows = []
    filing_sizes = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5),
                                 follow_redirects=True) as http:
        sec = SECClient(http, user_agent)
        for ticker, pinned in pinned_filings.items():
            ref = await sec.resolve_filing(ticker)
            if ref.accession != pinned:
                raise RuntimeError(f"Latest {ticker} accession changed; relabel before evaluation")
            filing = await sec.fetch_filing(ref)
            passages = make_passages(ref.accession, filing.sections)
            vectors = await model.embed([passage.text for passage in passages])
            indexed = [replace(passage, embedding=vector)
                       for passage, vector in zip(passages, vectors)]
            hybrid = HybridRetriever(model, indexed)
            filing_sizes[ticker] = {"accession": pinned, "url": ref.url,
                                    "passages": len(indexed)}
            for labelled_ticker, item, question, source_anchor in cases:
                if labelled_ticker != ticker:
                    continue
                expected = {passage.id for passage in indexed
                            if passage.item == item and re.search(source_anchor, passage.text, re.I)}
                if not expected:
                    raise ValueError(f"Source anchor for {ticker} {item} is absent from passages")
                ranked = await model.search(question, indexed, item, k=4)
                hybrid_ranked = await hybrid.search(question, item, k=4)

                async def embedding_search(query, search_item, k):
                    return await model.search(query, indexed, search_item, k=k)

                baseline_selected = await context_ids(embedding_search, item)
                selected = list(await select_passages(model, indexed, item, SECTIONS[item]))
                hybrid_selected = await context_ids(hybrid.search, item)
                rows.append({
                    "ticker": ticker, "item": item, "question": question,
                    "sourceAnchor": source_anchor, "expectedPassageIds": sorted(expected),
                    "top4Ids": [passage.id for passage in ranked],
                    "hitAt4": any(passage.id in expected for passage in ranked),
                    "baselineContextIds": baseline_selected,
                    "presentInBaselineContext": any(identifier in expected
                                                    for identifier in baseline_selected),
                    "productionContextIds": selected,
                    "presentInProductionContext": any(identifier in expected for identifier in selected),
                    "hybridTop4Ids": [passage.id for passage in hybrid_ranked],
                    "hybridHitAt4": any(passage.id in expected for passage in hybrid_ranked),
                    "hybridContextIds": hybrid_selected,
                    "presentInHybridContext": any(identifier in expected for identifier in hybrid_selected),
                })
    total = len(rows)
    return {
        "source": f"{dataset}: pinned live SEC filings and {len(cases)} manually labeled passage anchors",
        "model": "local MiniLM: frozen baseline vs diversified live context; dev-only BM25 experiment",
        "scope": "retrieval evidence only, no Groq calls or generated-answer accuracy claim",
        "filings": filing_sizes,
        "recallAt4": {"hits": sum(row["hitAt4"] for row in rows), "questions": total},
        "baselineContextCoverage": {
            "hits": sum(row["presentInBaselineContext"] for row in rows),
            "questions": total, "maxPassagesPerItem": 6,
        },
        "productionContextCoverage": {
            "hits": sum(row["presentInProductionContext"] for row in rows),
            "questions": total, "maxPassagesPerItem": 6,
        },
        "hybridRecallAt4": {
            "hits": sum(row["hybridHitAt4"] for row in rows), "questions": total,
        },
        "hybridProductionContextCoverage": {
            "hits": sum(row["presentInHybridContext"] for row in rows),
            "questions": total, "maxPassagesPerItem": 6,
        },
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("development", "holdout", "fresh"),
                        default="development")
    print(json.dumps(asyncio.run(evaluate(parser.parse_args().dataset)), indent=2))
