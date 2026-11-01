"""B3 混合检索：Reciprocal Rank Fusion（RRF）。

为什么用 RRF 而不是直接加权分数
------------------------------
BM25 的分数是"无上界的正数"（可能 0.3，也可能 30），稠密内积在 [-1, 1]。
两者量纲完全不同，做线性加权必须先把分数量纲对齐（min-max、z-score…），
而这类归一会随查询分布漂移——今天调好的权重，换个数据集就废。

RRF 只用**排名**，不用分数：

    RRF(d) = Σ_{r ∈ 各检索线} 1 / (k + rank_r(d))

rank 从 1 起算。这样每条线贡献被压缩到 (0, 1/(k+1)] 区间，量纲天然对齐，
无需任何调参即可跨数据集稳定工作。这是它成为工业默认做法的原因。

k 的作用
--------
k 越大，各排名的权重越接近（趋于平均）；k 越小，头部排名越被放大。
Cormack et al. 2009 提出时用 k=60，成了后续的默认值。
但"默认值合理"和"对本任务最优"是两件事——面试会被追问，所以要真扫一遍。

注意一个常见误读
----------------
RRF 融合的是**候选集合**。如果只取每条线的 top-10 融合，那 top-10 之外的
好文档永远进不来。本实现取每条线 top-100，融合后再截断到 top_k。
"""

from __future__ import annotations


def rrf_fuse(
    runs: list[dict[str, dict[str, float]]],
    k: int = 60,
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    """把多条线的检索结果按 RRF 融合。

    runs: [run1, run2, ...]，每个 run 是 {qid: {doc_id: score}}
    k:    RRF 平滑常数
    """
    if not runs:
        return {}

    qids = set()
    for r in runs:
        qids |= set(r.keys())

    fused: dict[str, dict[str, float]] = {}
    for qid in qids:
        acc: dict[str, float] = {}
        for run in runs:
            # 各线自己的排序，分数只用来定名次
            ranked = sorted(run.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (doc, _) in enumerate(ranked, start=1):
                acc[doc] = acc.get(doc, 0.0) + 1.0 / (k + rank)

        top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        fused[qid] = dict(top)

    return fused


def weighted_score_fuse(
    runs: list[dict[str, dict[str, float]]],
    weights: list[float],
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    """对照组：先 min-max 归一再加权求和。

    保留这个实现是为了在报告里对比「归一化加权 vs RRF」——
    面试常问"为什么不直接加权"，有这个对照才有话可说。
    """
    if len(runs) != len(weights):
        raise ValueError("runs 与 weights 数量不一致")

    qids = set()
    for r in runs:
        qids |= set(r.keys())

    fused: dict[str, dict[str, float]] = {}
    for qid in qids:
        acc: dict[str, float] = {}
        for run, w in zip(runs, weights):
            scores = run.get(qid, {})
            if not scores:
                continue
            lo, hi = min(scores.values()), max(scores.values())
            span = hi - lo if hi > lo else 1.0
            for doc, s in scores.items():
                acc[doc] = acc.get(doc, 0.0) + w * (s - lo) / span
        top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        fused[qid] = dict(top)

    return fused


if __name__ == "__main__":
    # 自检：两条线各把不同文档排第一，融合后两者都应进 top-2
    a = {"q": {"d1": 9.0, "d2": 5.0, "d3": 1.0}}
    b = {"q": {"d3": 0.9, "d1": 0.5, "d2": 0.1}}
    out = rrf_fuse([a, b], k=60, top_k=3)["q"]
    # d1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    # d3: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
    assert list(out)[0] == "d1", out
    assert abs(out["d1"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(out["d3"] - (1 / 63 + 1 / 61)) < 1e-12
    print("hybrid.py 自检通过", {k: round(v, 5) for k, v in out.items()})
