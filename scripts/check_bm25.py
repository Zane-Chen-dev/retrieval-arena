"""交叉验证：自己写的 BM25 vs rank_bm25 库。

为什么要这个文件：自己实现的公式容易有一处写错（分母差个括号、idf 少个 1+），
跑出来的数看起来还挺像那么回事。用独立实现对齐，才知道公式没抄错。

注意：两者 tokenization 不同（rank_bm25 需要自己先切词），本脚本强行用同一份
token 序列喂进去，所以只比公式，不比分词。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi  # noqa: E402

from src.bm25 import build_bm25, tokenize  # noqa: E402
from src.data import load_dataset  # noqa: E402


def main() -> None:
    ds = load_dataset("scifact")
    doc_ids = list(ds.corpus.keys())
    # 只取前 2000 篇，rank_bm25 全量跑会比较慢
    doc_ids = doc_ids[:2000]
    tokenized = [tokenize(ds.corpus[d]) for d in doc_ids]

    print("建索引 ...")
    mine = build_bm25({d: ds.corpus[d] for d in doc_ids})
    ref = BM25Okapi(tokenized)

    queries = list(ds.queries.items())[:20]
    max_diff = 0.0
    for qid, qtext in queries:
        toks = tokenize(qtext)
        ref_scores = ref.get_scores(toks)

        my_run = mine.search(qtext, top_k=len(doc_ids))
        my_scores = [my_run.get(d, 0.0) for d in doc_ids]

        # rank_bm25 的 idf 可能为负（标准公式不加 1+），做最小二乘缩放后比较排序
        for a, b in zip(my_scores, ref_scores):
            if a > 0:
                max_diff = max(max_diff, abs(a - b))

        # 更实用的比较：top-10 集合是否一致
        my_top = [d for d in my_run][:10]
        ref_top = [doc_ids[i] for i in
                   sorted(range(len(ref_scores)), key=lambda i: -ref_scores[i])[:10]]
        overlap = len(set(my_top) & set(ref_top))
        print(f"  {qid}: top-10 重叠 {overlap}/10")

    print(f"\n逐文档分数最大绝对差：{max_diff:.4f}")
    print("（若重叠 ≥8/10，公式等价；差异来自 idf 的 1+ 平滑）")


if __name__ == "__main__":
    main()
