"""Evaluate local embedding search on an invented, labeled passage collection.

This is a synthetic sanity check, not a measure of accuracy on real SEC filings.
Run: uv run python -m scripts.evaluate_retrieval
"""

import asyncio
import json
from dataclasses import replace

from retrieval import LocalEmbedder, Passage

CASES = [
    ("Item 1", "Which customers buy the company's sensors?", "1:0"),
    ("Item 1A", "What supplier risk could halt sensor production?", "1a:0"),
    ("Item 7", "What did management report about revenue growth?", "7:0"),
    ("Item 8", "What was net income?", "8:0"),
]
EXAMPLES = [
    ("Item 1", "1:0", "Acme sells industrial sensors to hospital customers and manufacturing customers across North America."),
    ("Item 1", "1:1", "Acme leases two factories and an administrative office near its headquarters."),
    ("Item 1A", "1a:0", "A single supplier could interrupt production of industrial sensors if it fails to deliver parts."),
    ("Item 1A", "1a:1", "Foreign currency exchange rates may affect overseas sales and costs."),
    ("Item 7", "7:0", "Management reported that revenue grew 8 percent this year due to higher sensor demand."),
    ("Item 7", "7:1", "Management recorded a restructuring charge for one-time office consolidation."),
    ("Item 8", "8:0", "The financial statements report revenue of 108 million dollars and net income of 12 million dollars."),
    ("Item 8", "8:1", "Cash and cash equivalents were 42 million dollars at year end."),
]


async def main():
    embedder = LocalEmbedder()
    passages = [Passage(identifier, "synthetic", item, index, text)
                for index, (item, identifier, text) in enumerate(EXAMPLES)]
    vectors = await embedder.embed([passage.text for passage in passages])
    indexed = [replace(passage, embedding=vector) for passage, vector in zip(passages, vectors)]
    rows = []
    for item, question, expected in CASES:
        ranked = await embedder.search(question, indexed, item, k=1)
        rows.append({"item": item, "query": question, "expected": expected,
                     "retrieved": [passage.id for passage in ranked],
                     "hit": any(passage.id == expected for passage in ranked)})
    hits = sum(row["hit"] for row in rows)
    print(json.dumps({"dataset": "synthetic hand-labeled examples, not real SEC evidence",
                      "retrieval": "local MiniLM, cosine, item-restricted, k=1",
                      "recallAt1": {"hits": hits, "questions": len(rows),
                                    "ratio": hits / len(rows)}, "cases": rows}, indent=2))
    if hits != len(rows):
        raise SystemExit("One or more labeled fixture passages were not retrieved at rank 1")


if __name__ == "__main__":
    asyncio.run(main())
