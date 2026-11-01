"""延迟测量。

为什么要预热：第一次查询会触发磁盘页加载、CPU 缓存填充、Python 首次字节码编译，
这些开销不代表稳态性能。不预热直接计时，P99 会被冷启动污染。

为什么用分位数而不是平均值：平均值会被长尾拉平，掩盖真正的抖动。
用户感知的是 P99，不是 mean。
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np


def measure(
    fn: Callable[[], object],
    n_warmup: int = 20,
    n_repeat: int = 200,
) -> dict[str, float]:
    """跑 n_warmup 次预热，再计时 n_repeat 次，返回毫秒分位数。"""
    for _ in range(n_warmup):
        fn()

    lat = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)

    arr = np.asarray(lat)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def measure_batch(
    queries: list[str],
    search_fn: Callable[[str], object],
    n_warmup: int = 20,
) -> dict[str, float]:
    """对一批 query 逐条计时（每次一条），预热用前 n_warmup 条循环两遍。"""
    if len(queries) < 2:
        raise ValueError("query 太少，无法测量")

    for _ in range(2):
        for q in queries[:n_warmup]:
            search_fn(q)

    lat = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        lat.append((time.perf_counter() - t0) * 1000)

    arr = np.asarray(lat)
    return {
        "n_queries": len(queries),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "median_of_means": float(statistics.median(arr.tolist())),
    }
