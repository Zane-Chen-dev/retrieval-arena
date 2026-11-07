"""离线复现 SciFact 三条基线并记录案例，不覆盖已有结果。"""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="新的 JSON 文件，不覆盖已有文件")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出已存在：{args.output}")
    dataset = ROOT / "data" / "scifact"
    model = ROOT / "models" / "all-MiniLM-L6-v2"
    cache = ROOT / "data" / "_emb_cache" / "emb_sentence-transformers_all-MiniLM-L6-v2_5183.npy"
    for path in (dataset / "corpus.jsonl", dataset / "queries.jsonl", dataset / "qrels" / "test.tsv", model / "modules.json", cache):
        if not path.is_file():
            raise FileNotFoundError(f"缺少离线资源：{path}")
    ds = load_dataset("scifact")
    ids = [q for q in ds.qrels if q in ds.queries]
    bm = build_bm25(ds.corpus)
    encoder, dense = build_dense(ds.corpus, model_name=DEFAULT_MODEL, M=32, ef_search=64, cache_dir=ROOT / "data" / "_emb_cache")
    vectors = encoder.encode([ds.queries[q] for q in ids], show_progress=False)
    runs = {"bm25": {}, "dense": {}, "rrf": {}}
    elapsed = {name: [] for name in runs}
    for i, qid in enumerate(ids):
        start = time.perf_counter()
        a = bm.search(ds.queries[qid], 100)
        t1 = time.perf_counter()
        b = dense.search(vectors[i], 100)
        t2 = time.perf_counter()
        c = rrf_fuse([{qid: a}, {qid: b}], k=60, top_k=100)[qid]
        t3 = time.perf_counter()
        for name, result in (("bm25", a), ("dense", b), ("rrf", c)):
            runs[name][qid] = result
        elapsed["bm25"].append((t1-start)*1000)
        elapsed["dense"].append((t2-t1)*1000)
        elapsed["rrf"].append((t3-start)*1000)
    qrels = {q: ds.qrels[q] for q in ids}
    cases = []
    for qid in ids:
        one = {qid: qrels[qid]}
        scores = {name: ndcg_at_k(run, one, 10) for name, run in runs.items()}
        rankings = {name: [{"doc_id": d, "relevance": qrels[qid].get(d, 0)} for d, _ in sorted(run[qid].items(), key=lambda x: (-x[1], x[0]))[:10]] for name, run in runs.items()}
        cases.append({"query_id": qid, "query": ds.queries[qid], "ndcg10": scores, "top10": rankings})
    cases.sort(key=lambda c: c["ndcg10"]["rrf"] - max(c["ndcg10"]["bm25"], c["ndcg10"]["dense"]))
    output = {"dataset": ds.stats, "judged_queries": len(ids), "parameters": {"top_k": 100, "rrf_k": 60, "M": 32, "ef_search_configured": 64, "ef_search_effective": 100, "gain": "trec"}, "metrics": {k: evaluate(v, qrels) for k, v in runs.items()}, "latency_ms": {k: {"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99))} for k, v in elapsed.items()}, "latency_scope": "按查询顺序单次计时，无预热；不含查询向量编码、索引构建和模型加载，非端到端延迟", "cases": {"fusion_worse": cases[:2], "fusion_better": cases[-2:]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"metrics": output["metrics"], "latency_ms": output["latency_ms"], "cases": output["cases"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
