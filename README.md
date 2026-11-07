# retrieval-arena

一个基于 BEIR 数据集的离线检索评测项目，比较 BM25、稠密向量检索和 RRF 混合检索的排序效果与检索耗时。项目用于学习和实验，不是线上服务或完整的 RAG 问答系统。

## 已实现与计划

- **B1：** 自实现倒排索引 BM25。
- **B2：** all-MiniLM-L6-v2 编码与 Faiss HNSW 检索。
- **B3：** BM25 与稠密检索结果的 RRF 融合，并提供归一化加权对照。
- **评测：** Recall@k、nDCG@k、MRR@k；支持 SciFact 和 NFCorpus。
- **实验性：** cross-encoder 重排代码及离线实验入口已加入；因模型暂不可用，尚未进行真实模型端到端验证。
- **未实现：** IVF-PQ、生成式问答、在线服务和增量索引。

## 环境与运行

需要 Python、pip，CPU 即可运行。以下示例使用 PowerShell；依赖包含 PyTorch 和 Faiss，请按操作系统及硬件情况确认安装兼容性。首次运行需要联网下载公开数据和模型；下载使用系统 TLS 证书校验。数据集和模型适用其各自的许可条款。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 首次运行自动下载数据；重复运行会覆盖对应的 results JSON
.\.venv\Scripts\python.exe scripts\run_bm25.py --dataset scifact

# 下载句向量模型，或通过 sentence-transformers 使用可访问的模型源
.\.venv\Scripts\python.exe scripts\download_model.py
.\.venv\Scripts\python.exe scripts\run_dense.py --dataset scifact
.\.venv\Scripts\python.exe scripts\run_hybrid.py --dataset scifact
```

`scripts/run_study.py --output <新的JSON路径>` 是离线实验入口，要求 SciFact、句向量模型和对应向量缓存均已存在，拒绝覆盖输出。已在本地现有环境验证 SciFact；结果包含融合效果改善和下降的案例。未标注文档的 `relevance` 为 `null`。

如遇证书错误，请检查系统信任链或网络环境；不要关闭证书验证。依赖尚未锁定，重新安装后结果及兼容性可能不同。

重排入口示例（需自行准备本地模型；当前尚未以真实模型验证）：

```powershell
.\.venv\Scripts\python.exe scripts\run_rerank.py --model models\ms-marco-MiniLM-L6-v2 --output results\rerank-new.json
```

该入口按 100 篇候选逐查询重排，未预热，计时不含查询编码和模型加载；小规模试运行可加 `--limit 5`，结果不代表全量数据集。实验和案例分析见 [实验记录](docs/实验记录.md)。

## 历史实验

以下来自仓库内实验结果，不代表所有环境都能复现。2026-11-07 在已有模型、缓存和环境下重新验证 SciFact 三条线；NFCorpus 数字为历史结果，尚待本轮复验。

| 数据集 / 方法 | nDCG@10 | Recall@100 |
| --- | ---: | ---: |
| SciFact / BM25 | 0.6645 | 0.8892 |
| SciFact / 稠密 HNSW | 0.6451 | 0.9250 |
| SciFact / RRF | 0.6853 | 0.9543 |
| NFCorpus / BM25 | 0.3117 | 0.2364 |
| NFCorpus / 稠密 HNSW | 0.3158 | 0.3112 |
| NFCorpus / RRF | 0.3443 | 0.3268 |

结果文件在 `results/`。相关文档以标注分数大于 0 判定；nDCG 默认使用 TREC gain（`2^rel - 1`）。未标注不等于一定不相关。历史延迟统计使用预先编码的查询向量，不包含查询编码、索引构建和模型加载，不能作为端到端服务延迟或工业性能结论。HNSW 在 top-k 为 100 时实际 efSearch 会被代码提升到至少 100。

## 结构与自检

- `src/data.py`：数据下载与加载。
- `src/bm25.py`、`src/dense.py`、`src/hybrid.py`：三条检索线。
- `src/evaluate.py`、`src/bench.py`：指标和耗时统计。
- `scripts/`：实验运行入口；`results/`：历史结果。

```powershell
.\.venv\Scripts\python.exe src\evaluate.py
.\.venv\Scripts\python.exe src\hybrid.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 许可

代码采用 [MIT License](LICENSE)。第三方数据、模型和依赖遵守各自的许可证。
