"""对已有候选按查询与文档的匹配分数重新排序。"""
from __future__ import annotations

import math
from collections.abc import Callable


def rerank(query: str, candidates: dict[str, float], corpus: dict[str, str],
           predict: Callable[[list[tuple[str, str]]], object]) -> dict[str, float]:
    """只重排已有候选；predict 返回与候选顺序一致的分数。"""
    ids = sorted(candidates, key=lambda d: (-candidates[d], d))
    if not ids:
        return {}
    pairs = [(query, corpus[d]) for d in ids]
    scores = list(predict(pairs))
    if len(scores) != len(ids):
        raise ValueError("重排模型返回的分数数量与候选数不一致")
    ranked = {d: float(s) for d, s in zip(ids, scores)}
    if any(not math.isfinite(s) for s in ranked.values()):
        raise ValueError("重排模型返回了非有限分数")
    return ranked
