"""B2 稠密基线：句向量编码 + Faiss HNSW。

两个关键设计点
--------------
1. **向量做 L2 归一化后再用内积**。归一化后内积 = 余弦相似度，而 HNSW 支持内积
   （`METRIC_INNER_PRODUCT`），检索到的就是"方向最接近"的文档。
   不归一化直接内积，长文档（向量模长大）会天然占便宜，和 BM25 的 b 参数要解决的问题同源。

2. **HNSW 是近似检索**。`efSearch` 控制搜索时考察的候选队列长度：
   - 调小 → 快，但可能漏掉真正的近邻（召回掉）
   - 调大 → 准，但每次查询要访问更多点（延迟涨）
   Day 10 的「召回—延迟」曲线就是扫这个参数扫出来的。

为什么用 HNSW 而不是暴力检索（Flat）：
    Flat 是精确的，5000 文档暴力算也就几毫秒，但它 O(n)，文档量一上来就废。
    HNSW 是 O(log n) 量级的图检索，是真正的工程选择。用 Flat 当 B2 基线，
    后面没法谈延迟。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

# 国内直连 huggingface.co 大概率失败，走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_model(name: str) -> str:
    """优先用本地已下载的模型目录，避免依赖 HF hub。

    国内环境先跑 `python scripts/download_model.py`，模型会落在 models/<短名>/。
    本地存在就直接返回路径，sentence_transformers 从路径加载不会联网。
    """
    short = name.split("/")[-1]
    local = PROJECT_ROOT / "models" / short
    if (local / "modules.json").exists():
        return str(local)
    return name



@dataclass
class DenseEncoder:
    """句向量编码器。"""

    model_name: str = DEFAULT_MODEL
    batch_size: int = 64
    device: str = "cpu"
    _model: object = field(default=None, repr=False)

    def load(self) -> "DenseEncoder":
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(resolve_model(self.model_name), device=self.device)
        return self

    @property
    def dim(self) -> int:
        if self._model is None:
            raise RuntimeError("先调用 load()")
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """返回 L2 归一化后的 float32 矩阵，shape = (n, dim)。"""
        if self._model is None:
            raise RuntimeError("先调用 load()")
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 归一化后内积 == 余弦
        )
        return np.ascontiguousarray(vecs.astype(np.float32))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text], show_progress=False)[0]


@dataclass
class HNSWRetriever:
    """Faiss HNSW 索引封装。"""

    dim: int
    M: int = 32                 # 每个节点的出边数，越大图越稠密、精度越高、内存越大
    ef_construction: int = 200  # 建图时考察的候选数，影响图质量（不占查询时间）
    ef_search: int = 64         # 查询时考察的候选数，这是 Day 10 要扫的参数
    doc_ids: list[str] = field(default_factory=list)
    index: object = field(default=None, repr=False)

    def build(self, embeddings: np.ndarray, doc_ids: list[str]) -> "HNSWRetriever":
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError("向量数与 doc_ids 数量不一致")
        self.doc_ids = list(doc_ids)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(embeddings)
        self.set_ef_search(self.ef_search)
        return self

    def set_ef_search(self, ef: int) -> None:
        """运行时可改，用于扫参数。注意 faiss 内部要求 ef >= k。"""
        self.ef_search = ef
        self.index.hnsw.efSearch = ef

    def search(self, qvec: np.ndarray, top_k: int = 100) -> dict[str, float]:
        """qvec 需已归一化。返回 {doc_id: score}，内积越大越相似。

        ⚠️ faiss 要求 efSearch >= top_k，否则拿不满结果。这里的处理是**就地抬高**
        efSearch 到 top_k——这会导致一个问题：如果你用 top_k=100 去扫
        efSearch ∈ {16, 32, 64}，三个点其实全跑在 100 上，扫出来是一条直线。

        所以做「召回—延迟」曲线时，**top_k 必须跟着一起降**（用 top_k=10 扫 ef），
        或者只扫 ef >= top_k 的区间。这不是 faiss 的毛病，是实验设计必须先想清楚的。
        """
        q = np.ascontiguousarray(qvec.reshape(1, -1).astype(np.float32))
        effective_ef = max(self.ef_search, top_k)
        if self.index.hnsw.efSearch != effective_ef:
            self.index.hnsw.efSearch = effective_ef
        dist, idx = self.index.search(q, top_k)
        out = {}
        for score, i in zip(dist[0], idx[0]):
            if i >= 0:  # faiss 用 -1 填充"不够 top_k 个结果"的位置
                out[self.doc_ids[i]] = float(score)
        return out

    def search_text(self, encoder: DenseEncoder, text: str, top_k: int = 100) -> dict[str, float]:
        return self.search(encoder.encode_one(text), top_k)

    @property
    def index_size_mb(self) -> float:
        """索引内存占用（MB）。HNSW 的图结构比原始向量大不少，这是它的隐性成本。"""
        return self.index.ntotal * self.dim * 4 / 1024 / 1024


def build_dense(
    corpus: dict[str, str],
    model_name: str = DEFAULT_MODEL,
    M: int = 32,
    ef_search: int = 64,
    cache_dir: Path | None = None,
) -> tuple[DenseEncoder, HNSWRetriever]:
    """编码全部文档并建 HNSW 索引。向量会缓存到磁盘，避免重复编码。"""
    encoder = DenseEncoder(model_name=model_name).load()
    doc_ids = list(corpus.keys())

    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"emb_{model_name.replace('/', '_')}_{len(doc_ids)}.npy"

    if cache is not None and cache.exists():
        print(f"      命中向量缓存 {cache.name}")
        emb = np.load(cache)
    else:
        print(f"      编码 {len(doc_ids)} 篇文档（{model_name}, dim={encoder.dim}）...")
        emb = encoder.encode([corpus[d] for d in doc_ids])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
            print(f"      向量已缓存到 {cache.name}")

    retriever = HNSWRetriever(dim=encoder.dim, M=M, ef_search=ef_search)
    retriever.build(emb, doc_ids)
    return encoder, retriever
