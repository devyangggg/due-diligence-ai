"""Offline comparison ranker; not used by the live agent.

The embedding-plus-BM25 experiment lost context coverage on independent SEC
filings, so it remains available only to reproduce the evaluation.
"""

import re

from rank_bm25 import BM25Okapi

from retrieval import LocalEmbedder, Passage

_LEXICAL_TOKENS = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "be", "by", "did", "do", "does", "for",
    "from", "how", "in", "is", "of", "on", "or", "the", "to", "was", "were",
    "what", "which", "with",
})
_RRF_K = 60


def _terms(text: str) -> list[str]:
    return [term for term in _LEXICAL_TOKENS.findall(text.lower()) if term not in _STOPWORDS]


class HybridRetriever:
    """Fuse local embedding and BM25 ranks within each filing item."""

    def __init__(self, embedder: LocalEmbedder, passages: list[Passage]):
        self.embedder = embedder
        self.items: dict[str, list[Passage]] = {}
        for passage in passages:
            self.items.setdefault(passage.item, []).append(passage)
        self.lexical = {
            item: BM25Okapi([_terms(passage.text) for passage in candidates])
            for item, candidates in self.items.items()
        }

    async def search(self, query: str, item: str, k: int = 2) -> list[Passage]:
        candidates = self.items.get(item, [])
        if not candidates or k <= 0:
            return []
        semantic = await self.embedder.search(query, candidates, item, k=len(candidates))
        fused = {passage.id: 1 / (_RRF_K + rank)
                 for rank, passage in enumerate(semantic, 1)}
        query_terms = _terms(query)
        if query_terms:
            lexical_scores = self.lexical[item].get_scores(query_terms)
            lexical = sorted(
                ((float(score), passage) for passage, score in zip(candidates, lexical_scores)
                 if score > 0),
                key=lambda row: (-row[0], row[1].index, row[1].id),
            )
            for rank, (_, passage) in enumerate(lexical, 1):
                fused[passage.id] += 1 / (_RRF_K + rank)
        return sorted(candidates, key=lambda passage: (
            -fused[passage.id], passage.index, passage.id
        ))[:k]
