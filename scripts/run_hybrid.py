"""B3：混合检索（BM25 + 稠密，RRF 融合）。

一张表同时给出 B1 / B2 / B3，直接看融合拿了多少回来。
附带扫 RRF 的 k ∈ {10, 60, 100}，回答面试必问的「k 为什么取 60」。

用法：
    .venv\\Scripts\\python.exe scripts\\run_hybrid.py --dataset scifact
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
from src.dense import DEFAULT_MODEL, build_dense  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.hybrid import rrf_fuse, weighted_score_fuse  # noqa: E402


def save(obj: dict, name: str) -> Path:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    fp = out / name
    fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp


def fmt(m: dict) -> str:
    return (f"nDCG@10={m['ndcg@10']:.4f}  R@100={m['recall@100']:.4f}  "
            f"MRR@10={m['mrr@10']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    print(f"[1/5] 加载 {args.dataset} ...")
    ds = load_dataset(args.dataset)
    print(f"      {ds.stats}")

    print("[2/5] B1 BM25 ...")
    t0 = time.perf_counter()
    bm25 = build_bm25(ds.corpus)
    run_b1 = bm25.search_batch(ds.queries, args.top_k)
    b1_ms = (time.perf_counter() - t0) * 1000

    print("[3/5] B2 稠密 ...")
    t0 = time.perf_counter()
    encoder, dense = build_dense(
        ds.corpus, model_name=args.model, M=args.M,
        ef_search=args.ef_search, cache_dir=ROOT / "data" / "_emb_cache",
    )
    b2_build_ms = (time.perf_counter() - t0) * 1000
    qids = list(ds.queries.keys())
    qvecs = encoder.encode([ds.queries[q] for q in qids], show_progress=False)
    run_b2 = {qid: dense.search(qvecs[i], args.top_k) for i, qid in enumerate(qids)}

    print(f"[4/5] B3 RRF 融合 (k={args.rrf_k}) ...")
    run_b3 = rrf_fuse([run_b1, run_b2], k=args.rrf_k, top_k=args.top_k)

    print("[5/5] 评测 ...")
    lines = {
        "B1-bm25": run_b1,
        "B2-dense": run_b2,
        f"B3-rrf{args.rrf_k}": run_b3,
    }
    results = {name: evaluate(run, ds.qrels, k_values=(10, 100)) for name, run in lines.items()}

    print("\n" + "=" * 74)
    print(f"  B1 / B2 / B3 对照 · {args.dataset}")
    print("=" * 74)
    print(f"  {'线':14s}{'nDCG@10':>10s}{'R@100':>10s}{'MRR@10':>10s}")
    print("-" * 74)
    for name, m in results.items():
        print(f"  {name:14s}{m['ndcg@10']:>10.4f}{m['recall@100']:>10.4f}{m['mrr@10']:>10.4f}")
    print("-" * 74)
    gain = results[f'B3-rrf{args.rrf_k}']["ndcg@10"] - max(
        results["B1-bm25"]["ndcg@10"], results["B2-dense"]["ndcg@10"]
    )
    print(f"  融合相对单路最好成绩： nDCG@10 {'+' if gain >= 0 else ''}{gain * 100:.2f} 个点")
    print("=" * 74)

    # 延迟：三条线各测一遍（融合线 = 两条线耗时之和 + 融合开销）
    print("\n  延迟（只算有标注的 query）...")
    judged = [q for q in qids if q in ds.qrels]
    judged_vecs = {q: qvecs[qids.index(q)] for q in judged}

    lat = {}
    lat["B1-bm25"] = measure_batch(judged, lambda q: bm25.search(ds.queries[q], args.top_k))
    lat["B2-dense"] = measure_batch(judged, lambda q: dense.search(judged_vecs[q], args.top_k))

    def b3_search(q: str) -> dict:
        r1 = bm25.search(ds.queries[q], args.top_k)
        r2 = dense.search(judged_vecs[q], args.top_k)
        return rrf_fuse([{q: r1}, {q: r2}], k=args.rrf_k, top_k=args.top_k)[q]

    lat["B3-rrf"] = measure_batch(judged, b3_search)
    for name, l in lat.items():
        print(f"    {name:10s} P50={l['p50_ms']:7.3f}ms  P99={l['p99_ms']:7.3f}ms")

    # 附加：RRF k 敏感性（面试 Q2 的直接证据）
    print("\n  [附加] RRF k 敏感性 ...")
    k_sweep = []
    for k in (10, 60, 100):
        r = rrf_fuse([run_b1, run_b2], k=k, top_k=args.top_k)
        m = evaluate(r, ds.qrels, k_values=(10, 100))
        k_sweep.append({"k": k, **m})
        print(f"    k={k:4d}  nDCG@10={m['ndcg@10']:.4f}  R@100={m['recall@100']:.4f}")

    # 附加：归一化加权 vs RRF（面试"为什么不直接加权"的对照）
    print("\n  [附加] 归一化加权融合 vs RRF ...")
    w_fused = weighted_score_fuse([run_b1, run_b2], [0.5, 0.5], top_k=args.top_k)
    wm = evaluate(w_fused, ds.qrels, k_values=(10, 100))
    print(f"    加权平均(0.5/0.5)  nDCG@10={wm['ndcg@10']:.4f}")
    print(f"    RRF(k={args.rrf_k})           nDCG@10={results[f'B3-rrf{args.rrf_k}']['ndcg@10']:.4f}")

    fp = save(
        {
            "dataset": args.dataset,
            "params": {"model": args.model, "M": args.M, "ef_search": args.ef_search,
                       "top_k": args.top_k, "rrf_k": args.rrf_k},
            "results": results,
            "latency": lat,
            "rrf_k_sweep": k_sweep,
            "weighted_fuse_0.5": wm,
            "b1_build_ms": b1_ms,
            "b2_build_ms": b2_build_ms,
        },
        f"b3_hybrid_{args.dataset}.json",
    )
    print(f"\n  已保存 {fp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
