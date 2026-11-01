"""Day 1-4 目标：跑出 BM25 基线，拿到第一个数。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe scripts\\run_bm25.py
    .venv\\Scripts\\python.exe scripts\\run_bm25.py --dataset nfcorpus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bench import measure_batch  # noqa: E402
from src.bm25 import build_bm25  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.evaluate import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    print(f"[1/4] 加载 {args.dataset} ...")
    ds = load_dataset(args.dataset)
    print(f"      {ds.stats}")

    print(f"[2/4] 建 BM25 索引 (k1={args.k1}, b={args.b}) ...")
    t0 = time.perf_counter()
    index = build_bm25(ds.corpus, k1=args.k1, b=args.b)
    build_s = time.perf_counter() - t0
    print(f"      耗时 {build_s:.2f}s，词表 {len(index.idf)} 词，平均文档长度 {index.avgdl:.1f} 词")

    print("[3/4] 全部 query 检索 top-100 ...")
    run = {}
    for i, (qid, qtext) in enumerate(ds.queries.items(), 1):
        run[qid] = index.search(qtext, args.top_k)
        if i % 200 == 0:
            print(f"      {i}/{len(ds.queries)}")

    print("[4/4] 评测 ...")
    metrics = evaluate(run, ds.qrels, k_values=(10, 100))

    # 只对"有标注"的 query 算延迟，否则无标注 query 的空结果会拉低延迟
    judged = [q for q in ds.queries if q in ds.qrels]
    lat = measure_batch([ds.queries[q] for q in judged], lambda q: index.search(q, args.top_k))

    print("\n" + "=" * 58)
    print(f"  B1 稀疏基线 · BM25 · {args.dataset}")
    print("=" * 58)
    for k, v in metrics.items():
        print(f"  {k:14s} {v:.4f}")
    print("-" * 58)
    print(f"  索引构建        {build_s:.2f} s")
    print(f"  查询 P50        {lat['p50_ms']:.3f} ms")
    print(f"  查询 P95        {lat['p95_ms']:.3f} ms")
    print(f"  查询 P99        {lat['p99_ms']:.3f} ms")
    print("=" * 58)

    out = {
        "line": "B1-bm25",
        "dataset": args.dataset,
        "params": {"k1": args.k1, "b": args.b, "top_k": args.top_k},
        "metrics": metrics,
        "latency": lat,
        "index_build_s": build_s,
        "n_docs": len(ds.corpus),
        "n_queries": len(ds.queries),
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    fp = out_dir / f"b1_bm25_{args.dataset}.json"
    fp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  已保存 {fp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
