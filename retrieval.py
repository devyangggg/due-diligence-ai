"""Deterministic filing passages and local-only embedding-based retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import threading
from typing import Any

MODEL_VERSION = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKER_VERSION = "paragraph-v1"
_MIN_TOKENS = 250
_TARGET_TOKENS = 300
_MAX_TOKENS = 350


@dataclass(frozen=True)
class Passage:
    id: str
    accession: str
    item: str
    index: int
    text: str
    embedding: bytes | None = None


def _paragraph_endpoints(text: str) -> list[int]:
    """Character offsets at paragraph, sentence and hard token boundaries."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    boundaries: list[int] = []
    for index, word in enumerate(words, 1):
        next_start = words[index].start() if index < len(words) else len(text)
        gap = text[word.end():next_start]
        if index == len(words) or "\n\n" in gap:
            boundaries.append(index)
    return boundaries


def make_passages(accession: str, sections: dict[str, str]) -> list[Passage]:
    """Split on paragraphs where possible, preserving exact contiguous source slices.

    Long paragraphs are broken at sentence endings when possible, then at a
    bounded whitespace token boundary. Passage IDs are stable for the same
    accession, item and chunker version.
    """
    passages: list[Passage] = []
    for item, source in sections.items():
        text = source.strip()
        words = list(re.finditer(r"\S+", text))
        if not words:
            continue
        paragraphs = set(_paragraph_endpoints(text))
        sentences = {index for index, word in enumerate(words, 1)
                     if re.search(r"[.!?][\"'\u2019\u201d)]*$", word.group())}
        start = 0
        chunk_index = 0
        while start < len(words):
            maximum = min(start + _MAX_TOKENS, len(words))
            if maximum == len(words):
                end = maximum
            else:
                minimum = min(start + _MIN_TOKENS, maximum)
                candidates = [boundary for boundary in paragraphs
                              if minimum <= boundary <= maximum]
                if not candidates:
                    candidates = [boundary for boundary in sentences
                                  if minimum <= boundary <= maximum]
                end = min(candidates, key=lambda candidate: (abs(candidate - start - _TARGET_TOKENS), -candidate)) if candidates else maximum
            left = words[start].start()
            right = words[end - 1].end()
            # Include intervening paragraph whitespace within the same chunk;
            # whitespace between chunks is immaterial, never rewritten.
            excerpt = text[left:right].strip()
            passages.append(Passage(id=f"{accession}:{item}:{chunk_index}", accession=accession,
                                    item=item, index=chunk_index, text=excerpt))
            chunk_index += 1
            start = end
    return passages


class LocalEmbedder:
    """One lazily loaded sentence-transformers model, encoded off the event loop."""

    _model: Any = None
    _model_lock = threading.Lock()

    @classmethod
    def _encode_sync(cls, texts: list[str]) -> list[bytes]:
        with cls._model_lock:
            if cls._model is None:
                from sentence_transformers import SentenceTransformer

                cls._model = SentenceTransformer(MODEL_VERSION)
            import numpy as np

            vectors = cls._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            result: list[bytes] = []
            for vector in vectors:
                row = np.asarray(vector, dtype="<f4")
                norm = np.linalg.norm(row)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("Embedding model returned a non-finite or zero vector")
                row = row / norm
                result.append(np.asarray(row, dtype="<f4").tobytes())
            return result

    async def embed(self, texts: list[str]) -> list[bytes]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts)

    async def search(self, query: str, passages: list[Passage], item: str,
                     k: int = 4) -> list[Passage]:
        if k <= 0:
            return []
        candidates = [passage for passage in passages if passage.item == item]
        if not candidates:
            return []
        if any(passage.embedding is None for passage in candidates):
            raise ValueError(f"Missing embeddings for {item}")
        import numpy as np

        query_vector = np.frombuffer((await self.embed([query]))[0], dtype="<f4")
        scores: list[tuple[float, Passage]] = []
        for passage in candidates:
            vector = np.frombuffer(passage.embedding, dtype="<f4")
            if vector.size != query_vector.size:
                raise ValueError(f"Incompatible embedding dimension for passage {passage.id}")
            scores.append((float(np.dot(query_vector, vector)), passage))
        scores.sort(key=lambda row: (-row[0], row[1].index, row[1].id))
        return [passage for _, passage in scores[:k]]


