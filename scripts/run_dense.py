"""B2：稠密检索基线（all-MiniLM-L6-v2 + Faiss HNSW）。

用法：
    .venv\\Scripts\\python.exe scripts\\run_dense.py --dataset scifact
    .venv\\Scripts\\python.exe scripts\\run_dense.py --dataset scifact --sweep-ef
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
from src.data import load_dataset  # noqa: E402
from src.dense import DEFAULT_MODEL, HNSWRetriever, build_dense  # noqa: E402
from src.evaluate import evaluate  # noqa: E402


def save(obj: dict, name: str) -> Path:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    fp = out_dir / name
    fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--sweep-ef", action="store_true", help="扫 efSearch 出召回-延迟曲线")
    ap.add_argument("--sweep-k", type=int, default=10,
                    help="扫参时使用的 top_k，必须 <= 最小 ef（否则曲线被 faiss 拉平）")
    args = ap.parse_args()

    print(f"[1/4] 加载 {args.dataset} ...")
    ds = load_dataset(args.dataset)
    print(f"      {ds.stats}")

    print("[2/4] 编码 + 建 HNSW 索引 ...")
    t0 = time.perf_counter()
    encoder, retr = build_dense(
        ds.corpus,
        model_name=args.model,
        M=args.M,
        ef_search=args.ef_search,
        cache_dir=ROOT / "data" / "_emb_cache",
    )
    build_s = time.perf_counter() - t0
    print(f"      耗时 {build_s:.1f}s，向量 {retr.index.ntotal} 条 × {retr.dim} 维")

    print(f"[3/4] 检索全部 query（efSearch={args.ef_search}）...")
    qids = list(ds.queries.keys())
    texts = [ds.queries[q] for q in qids]
    qvecs = encoder.encode(texts, show_progress=False)
    run = {
        qid: retr.search(qvecs[i], args.top_k)
        for i, qid in enumerate(qids)
    }

    print("[4/4] 评测 ...")
    metrics = evaluate(run, ds.qrels, k_values=(10, 100))

    judged = [q for q in qids if q in ds.qrels]
    judged_vecs = {q: qvecs[qids.index(q)] for q in judged}
    lat = measure_batch(
        judged, lambda q: retr.search(judged_vecs[q], args.top_k), n_warmup=20
    )

    print("\n" + "=" * 58)
    print(f"  B2 稠密基线 · {Path(args.model).name} + HNSW · {args.dataset}")
    print("=" * 58)
    for k, v in metrics.items():
        print(f"  {k:14s} {v:.4f}")
    print("-" * 58)
    print(f"  建索引+编码     {build_s:.1f} s")
    print(f"  efSearch        {args.ef_search}")
    print(f"  查询 P50        {lat['p50_ms']:.3f} ms")
    print(f"  查询 P95        {lat['p95_ms']:.3f} ms")
    print(f"  查询 P99        {lat['p99_ms']:.3f} ms")
    print(f"  向量内存        {retr.index_size_mb:.1f} MB（不含图结构）")
    print("=" * 58)

    fp = save(
        {
            "line": "B2-dense",
            "dataset": args.dataset,
            "params": {"model": args.model, "M": args.M, "ef_search": args.ef_search,
                       "top_k": args.top_k},
            "metrics": metrics,
            "latency": lat,
            "index_build_s": build_s,
            "index_size_mb": retr.index_size_mb,
        },
        f"b2_dense_{args.dataset}_ef{args.ef_search}.json",
    )
    print(f"  已保存 {fp.relative_to(ROOT)}")

    if args.sweep_ef:
        print("\n[附加] 扫 efSearch 出召回-延迟曲线 ...")
        print("      ⚠️ 扫参的 top_k 必须 <= 最小 ef，否则 faiss 会把 ef 抬到 top_k，曲线拉平。")
        sweep_k = args.sweep_k
        ef_list = [16, 32, 64, 128, 256]
        assert sweep_k <= min(ef_list), f"sweep_k={sweep_k} 大于最小 ef={min(ef_list)}"
        sweep = []
        for ef in ef_list:
            retr.set_ef_search(ef)
            r = {qid: retr.search(qvecs[i], sweep_k) for i, qid in enumerate(qids)}
            m = evaluate(r, ds.qrels, k_values=(sweep_k,))
            l = measure_batch(judged, lambda q: retr.search(judged_vecs[q], sweep_k),
                              n_warmup=10)
            sweep.append({"ef_search": ef, "top_k": sweep_k, **m,
                          "p99_ms": l["p99_ms"], "p50_ms": l["p50_ms"]})
            print(f"      ef={ef:4d}  recall@{sweep_k}={m[f'recall@{sweep_k}']:.4f}  "
                  f"ndcg@{sweep_k}={m[f'ndcg@{sweep_k}']:.4f}  "
                  f"P50={l['p50_ms']:.3f}ms  P99={l['p99_ms']:.3f}ms")

        fp2 = save({"dataset": args.dataset, "model": args.model,
                    "sweep_top_k": sweep_k, "sweep": sweep},
                   f"b2_dense_sweep_{args.dataset}.json")
        print(f"  已保存 {fp2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
