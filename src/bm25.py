"""BM25 倒排索引（B1 稀疏基线）。

为什么自己写而不用 `rank_bm25`：
1. 它是 `rank_bm25` 的 Okapi BM25，但**没有倒排**，每次查询都要遍历全部文档，
   5000 文档 × 1100 query 会很慢；本文件用倒排，只碰命中文档。
2. 面试会问「BM25 的公式和两个参数是干什么的」，自己写过才答得上。
   `rank_bm25` 保留在环境里做交叉验证（见 `scripts/check_bm25.py`）。

公式
----
    score(D, Q) = Σ_{t∈Q} IDF(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 - b + b·|D|/avgdl) )

    IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )     # 恒正，避免高频词得负分

参数含义
--------
    k1 = 1.5  控制词频饱和速度。k1 越大，同一个词重复出现越"值钱"；
              k1=0 时退化成二值（出现即 1，不看出几次）。
    b  = 0.75 控制文档长度归一化强度。b=0 完全不归一化（长文档天然占便宜），
              b=1 完全按长度惩罚。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """小写 + 按非字母数字切分。BEIR 的 BM25 基线基本就用这套。"""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    doc_len: list[int] = field(default_factory=list)
    # term -> [(doc_index, term_freq), ...]
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    idf: dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0

    def build(self, corpus: dict[str, str]) -> "BM25Index":
        self.doc_ids = list(corpus.keys())
        self.postings = defaultdict(list)
        self.doc_len = []

        for idx, did in enumerate(self.doc_ids):
            toks = tokenize(corpus[did])
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((idx, tf))

        n = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / n if n else 0.0
        for term, plist in self.postings.items():
            df = len(plist)
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, top_k: int = 100) -> dict[str, float]:
        """返回 {doc_id: score}，只含得分 > 0 的文档，最多 top_k 个。"""
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl

        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in self.postings[term]:
                denom = tf + k1 * (1 - b + b * self.doc_len[idx] / avgdl)
                scores[idx] += idf * tf * (k1 + 1) / denom

        if not scores:
            return {}
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return {self.doc_ids[idx]: s for idx, s in top}

    def search_batch(self, queries: dict[str, str], top_k: int = 100) -> dict[str, dict[str, float]]:
        return {qid: self.search(q, top_k) for qid, q in queries.items()}

    def reset(self) -> "BM25Index":
        self.doc_ids, self.doc_len, self.avgdl = [], [], 0.0
        self.postings = defaultdict(list)
        self.idf = {}
        return self


def build_bm25(corpus: dict[str, str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    return BM25Index(k1=k1, b=b).build(corpus)
