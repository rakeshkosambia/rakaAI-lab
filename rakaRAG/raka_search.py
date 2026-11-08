"""Search indexes shared by the chapters: BM25 keyword search and fake-embedding vector search.

Both index "documents" with an id and a text. From r02 on, a document can be a chunk of a real document.
Pure standard library.
"""
from __future__ import annotations

import math
from collections import Counter

import raka_common as rc


class BM25Index:
    """Okapi BM25 over a list of rc.Doc-like items (id, text)."""

    def __init__(self, docs: list[rc.Doc], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [d.id for d in docs]
        self.tf = [Counter(rc.tokenize(d.text)) for d in docs]
        self.length = [sum(c.values()) for c in self.tf]
        self.avg_length = sum(self.length) / len(self.length) if self.length else 0.0
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = rc.tokenize(query)
        out = []
        for counts, length in zip(self.tf, self.length):
            score = 0.0
            for term in terms:
                f = counts.get(term, 0)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                score += self.idf[term] * f * (self.k1 + 1) / norm
            out.append(score)
        return out

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        scored = [(doc_id, s) for doc_id, s in zip(self.ids, self.scores(query)) if s > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]


class VectorIndex:
    """Cosine search over hashing-embedder vectors. Ties break on id so runs repeat exactly."""

    def __init__(self, docs: list[rc.Doc], dim: int = 512):
        self.dim = dim
        self.ids = [d.id for d in docs]
        self.vectors = [rc.hash_embed(d.text, dim) for d in docs]

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = rc.hash_embed(query, self.dim)
        scored = [(doc_id, rc.cosine(q, v)) for doc_id, v in zip(self.ids, self.vectors)]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [item for item in scored if item[1] > 0][:k]


def full_ranking(index, docs: list[rc.Doc]):
    """Wrap an index as search(question) -> every id, best first (unmatched items last, by id)."""
    all_ids = [d.id for d in docs]

    def search(question: str) -> list[str]:
        hits = [doc_id for doc_id, _ in index.search(question, k=len(all_ids))]
        return hits + [i for i in all_ids if i not in hits]

    return search
