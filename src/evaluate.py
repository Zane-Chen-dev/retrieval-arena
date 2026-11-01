"""检索评测指标：Recall@k / nDCG@k / MRR@k。

参数约定
-------
run    : dict[qid] -> dict[doc_id] -> score      （检索系统输出，未排序）
qrels  : dict[qid] -> dict[doc_id] -> relevance  （0/1 或 0/1/2 分级标注）

两个容易出错的点
----------------
1. **relevance > 0 才算相关**。BEIR 的 qrels 里 0 表示"判定为不相关"，
   如果拿 `doc_id in qrels[qid]` 判断相关，会把不相关文档也算进去。
2. **gain 有两种口径**：
   - `raw`：gain = relevance            —— BEIR 官方 leaderboard 用的口径
   - `trec`：gain = 2**relevance - 1    —— TREC 标准口径，放大大相关文档的权重
   0/1 标注下两者完全等价（SciFact 就是 0/1）；分级标注下（NFCorpus 有 0/1/2）会不同。
   跑出来的数字要标清楚用的哪个口径，否则和论文对不上。
"""

from __future__ import annotations

import math
from typing import Literal

GainMode = Literal["raw", "trec"]


def _ranked_docs(run_q: dict[str, float], k: int | None = None) -> list[str]:
    """按分数降序返回文档 id。分数相同时按 doc_id 排序，保证可复现。"""
    ordered = sorted(run_q.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in ordered] if k is None else [d for d, _ in ordered[:k]]


def _gain(rel: int, mode: GainMode) -> float:
    return float(rel) if mode == "raw" else float(2 ** rel - 1)


def recall_at_k(run: dict, qrels: dict, k: int) -> float:
    """| retrieved_topk ∩ relevant | / | relevant |，对所有有标注的 query 取平均。"""
    scores = []
    for qid, rel_map in qrels.items():
        relevant = {d for d, r in rel_map.items() if r > 0}
        if not relevant:
            continue
        topk = set(_ranked_docs(run.get(qid, {}), k))
        scores.append(len(topk & relevant) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(run: dict, qrels: dict, k: int, gain_mode: GainMode = "trec") -> float:
    """DCG@k / IDCG@k，折损用 log2(rank+1)（rank 从 1 起算）。"""
    scores = []
    for qid, rel_map in qrels.items():
        if not any(r > 0 for r in rel_map.values()):
            continue

        dcg = 0.0
        for i, doc in enumerate(_ranked_docs(run.get(qid, {}), k)):
            dcg += _gain(rel_map.get(doc, 0), gain_mode) / math.log2(i + 2)

        ideal = sorted((_gain(r, gain_mode) for r in rel_map.values() if r > 0), reverse=True)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))

        if idcg > 0:
            scores.append(dcg / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(run: dict, qrels: dict, k: int) -> float:
    """第一个相关文档的排名倒数的平均值。"""
    scores = []
    for qid, rel_map in qrels.items():
        relevant = {d for d, r in rel_map.items() if r > 0}
        if not relevant:
            continue
        for rank, doc in enumerate(_ranked_docs(run.get(qid, {}), k), start=1):
            if doc in relevant:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(
    run: dict,
    qrels: dict,
    k_values: tuple[int, ...] = (10, 100),
    gain_mode: GainMode = "trec",
) -> dict[str, float]:
    """一次算齐 recall@k / nDCG@k / MRR@k。返回 {指标名: 值}。"""
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(run, qrels, k)
        out[f"ndcg@{k}"] = ndcg_at_k(run, qrels, k, gain_mode)
        out[f"mrr@{k}"] = mrr_at_k(run, qrels, k)
    return out


if __name__ == "__main__":
    # 自检：手算可核对的极小例子
    r = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}
    q = {"q1": {"a": 1, "c": 1}}
    assert recall_at_k(r, q, 2) == 0.5, recall_at_k(r, q, 2)      # top2 = a,b，命中 1/2
    assert recall_at_k(r, q, 3) == 1.0                            # top3 全中
    assert abs(mrr_at_k(r, q, 3) - 1.0) < 1e-12                   # a 排第 1
    # nDCG@2：dcg = 1/log2(2) = 1；idcg = 1 + 1/log2(3) = 1.6309
    assert abs(ndcg_at_k(r, q, 2) - 1 / (1 + 1 / math.log2(3))) < 1e-12
    # 零相关文档不应参与平均
    assert recall_at_k(r, {"q1": {"a": 1}, "q2": {"z": 0}}, 3) == 1.0
    print("evaluate.py 自检通过")
