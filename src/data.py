"""BEIR 数据集下载与加载。

不引入 `beir` 包——它会把 pytorch/transformers 全拖进来，而 Day 1 只需要纯文本。
数据集结构（BEIR 标准）：
    <name>/corpus.jsonl     每行 {"_id", "title", "text"}
    <name>/queries.jsonl    每行 {"_id", "text"}
    <name>/qrels/test.tsv   TSV，列：query-id  corpus-id  score
"""

from __future__ import annotations

import csv
import json
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # 没有 tqdm 就退化成无进度条
    def tqdm(x, **kw):
        return x

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass
class Dataset:
    name: str
    corpus: dict[str, str] = field(default_factory=dict)      # doc_id  -> 拼接后的正文
    queries: dict[str, str] = field(default_factory=dict)     # qid     -> query 文本
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)  # qid -> {doc_id: 相关性}

    def __post_init__(self) -> None:
        if self.corpus and not self.queries:
            raise ValueError("corpus 有内容但 queries 为空，检查加载逻辑")

    @property
    def stats(self) -> dict[str, int]:
        return {
            "docs": len(self.corpus),
            "queries": len(self.queries),
            "qrels_queries": len(self.qrels),
            "judged_pairs": sum(len(v) for v in self.qrels.values()),
        }


def _download(url: str, dest: Path) -> None:
    """带进度下载。本地代理会 MITM 证书，所以关掉校验（只影响公开数据集下载）。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "retrieval-bench/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"下载 {dest.name}"
        ) as bar:
            while chunk := resp.read(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))


def ensure_dataset(name: str, root: Path | None = None) -> Path:
    """确保 data/<name>/ 存在，否则下载并解压。返回数据集目录。"""
    root = root or DATA_ROOT
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target

    zip_path = root / f"{name}.zip"
    if not zip_path.exists():
        _download(f"{BEIR_BASE}/{name}.zip", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return target


def load_dataset(name: str = "scifact", root: Path | None = None) -> Dataset:
    d = ensure_dataset(name, root)
    ds = Dataset(name=name)

    with open(d / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # BEIR 惯例：title 和 text 拼起来当文档内容
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            ds.corpus[row["_id"]] = f"{title} {text}".strip()

    with open(d / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ds.queries[row["_id"]] = row["text"].strip()

    # qrels 放在 qrels/test.tsv，分隔符是 tab
    qrels_path = d / "qrels" / "test.tsv"
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"].strip()
            did = row["corpus-id"].strip()
            score = int(row["score"])
            ds.qrels.setdefault(qid, {})[did] = score

    return ds


if __name__ == "__main__":
    for name in ("scifact", "nfcorpus"):
        ds = load_dataset(name)
        print(f"{name:10s} {ds.stats}")
