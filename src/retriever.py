"""Hybrid retriever.

Two signals are fused with reciprocal rank fusion (RRF):
  * BM25 over word tokens, which captures vocabulary ("denominator", "bracket").
  * TF-IDF over character n-grams, which captures maths notation ("1/4", "x^2", "-(")
    that word tokenisers break apart.

The design keeps the project runnable offline. A dense embedding model can be added as
a third ranked list in `search` without changing any calling code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN = re.compile(r"\d+/\d+|\d+\.\d+|\d+|[a-z]+|\^|[+\-x*/=]")
STOP = {"the", "a", "an", "of", "is", "to", "and", "in", "it", "by", "as", "for", "on", "be", "or", "that", "this", "with", "what"}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Hit:
    id: str
    score: float      # fused RRF score, used for ranking
    lexical: float    # character n-gram cosine similarity
    keyword: float    # BM25 score: zero means no shared vocabulary at all
    payload: dict


class HybridIndex:
    def __init__(self, docs: list[dict], text_fn, id_key: str = "id", rrf_k: int = 60):
        if not docs:
            raise ValueError("HybridIndex needs at least one document")
        self.docs = docs
        self.ids = [d[id_key] for d in docs]
        self.texts = [text_fn(d) for d in docs]
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in self.texts])
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.matrix = self.tfidf.fit_transform(self.texts)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        bm = np.asarray(self.bm25.get_scores(tokenize(query) or ["_"]))
        cos = cosine_similarity(self.tfidf.transform([query]), self.matrix).ravel()
        fused = np.zeros(len(self.docs))
        for scores in (bm, cos):
            order = np.argsort(-scores)
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (self.rrf_k + rank + 1)
        top = np.argsort(-fused)[:k]
        return [Hit(self.ids[i], float(fused[i]), float(cos[i]), float(bm[i]), self.docs[i]) for i in top]
