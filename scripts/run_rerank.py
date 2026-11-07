"""离线比较 RRF 与 cross-encoder 重排；模型需预先放到本地。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from src.bm25 import build_bm25
from src.data import load_dataset
from src.dense import DEFAULT_MODEL, build_dense
from src.evaluate import evaluate, ndcg_at_k
from src.hybrid import rrf_fuse
from src.rerank import rerank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="本地 cross-encoder 模型目录")
    parser.add_argument("--output", type=Path, required=True, help="新的结果文件")
    parser.add_argument("--limit", type=int, default=0, help="最多查询数；0 表示所有已标注查询")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 必须大于等于 0")
    if args.output.exists():
        raise FileExistsError(args.output)
    required = [args.model / "config.json", args.model / "model.safetensors",
                ROOT / "models" / "all-MiniLM-L6-v2" / "modules.json",
                ROOT / "data" / "scifact" / "corpus.jsonl",
                ROOT / "data" / "scifact" / "queries.jsonl",
                ROOT / "data" / "scifact" / "qrels" / "test.tsv",
                ROOT / "data" / "_emb_cache" / "emb_sentence-transformers_all-MiniLM-L6-v2_5183.npy"]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"离线资源不存在：{p}")
    from sentence_transformers import CrossEncoder

    ds = load_dataset("scifact")
    ids = [q for q in ds.qrels if q in ds.queries]
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("没有已标注查询")
    bm = build_bm25(ds.corpus)
    enc, dense = build_dense(ds.corpus, model_name=DEFAULT_MODEL, cache_dir=ROOT / "data" / "_emb_cache")
    vecs = enc.encode([ds.queries[q] for q in ids], show_progress=False)
    cross = CrossEncoder(str(args.model.resolve()), device="cpu", local_files_only=True)
    before, after = {}, {}
    baseline_ms, rerank_ms = [], []
    for i, q in enumerate(ids):
        start = time.perf_counter()
        sparse = bm.search(ds.queries[q], 100)
        dense_result = dense.search(vecs[i], 100)
        fused = rrf_fuse([{q: sparse}, {q: dense_result}], k=60, top_k=100)[q]
        mid = time.perf_counter()
        reranked = rerank(ds.queries[q], fused, ds.corpus,
                          lambda pairs: cross.predict(pairs, show_progress_bar=False))
        end = time.perf_counter()
        before[q], after[q] = fused, reranked
        baseline_ms.append((mid - start) * 1000)
        rerank_ms.append((end - mid) * 1000)
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(ids)}", flush=True)
    qrels = {q: ds.qrels[q] for q in ids}
    cases = []
    for q in ids:
        rels = {q: qrels[q]}
        b, a = ndcg_at_k(before, rels, 10), ndcg_at_k(after, rels, 10)
        cases.append({"query_id": q, "query": ds.queries[q], "before_ndcg10": b,
                      "after_ndcg10": a, "relevant_doc_ids": [d for d, r in qrels[q].items() if r > 0],
                      "before_top10": sorted(before[q], key=lambda d: (-before[q][d], d))[:10],
                      "after_top10": sorted(after[q], key=lambda d: (-after[q][d], d))[:10]})
    cases.sort(key=lambda c: c["after_ndcg10"] - c["before_ndcg10"])
    stats = lambda v: {"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99))}
    output = {"dataset": "scifact", "n_queries": len(ids), "top_k": 100, "rrf_k": 60,
              "model": args.model.name, "gain": "trec", "metrics": {"rrf": evaluate(before, qrels), "reranked": evaluate(after, qrels)},
              "latency_ms": {"rrf": stats(baseline_ms), "rerank_only": stats(rerank_ms),
                             "rrf_plus_rerank": stats([a+b for a, b in zip(baseline_ms, rerank_ms)])},
              "latency_scope": "逐查询单次计时，无预热；排除模型加载、索引构建及查询编码；非端到端延迟",
              "cases": {"decreased": cases[:2], "increased": cases[-2:]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"metrics": output["metrics"], "latency_ms": output["latency_ms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
