"""从 ModelScope 下载 sentence-transformers 模型到本地。

为什么要这个脚本
----------------
通过 ModelScope 拉到本地目录，再让
`sentence_transformers` 从路径加载，绕开 hub 的下载逻辑。

下载完成后：
    SentenceTransformer("models/all-MiniLM-L6-v2")
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELSCOPE_API = "https://www.modelscope.cn/api/v1/models"
DEFAULT_REPO = "AI-ModelScope/all-MiniLM-L6-v2"

# SentenceTransformer 加载必需的文件
NEEDED = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
]


def _ctx() -> ssl.SSLContext:
    return ssl.create_default_context()


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "retrieval-arena/0.1"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        return r.read()


def download(repo: str = DEFAULT_REPO, dest: Path | None = None) -> Path:
    short = repo.split("/")[-1]
    dest = dest or ROOT / "models" / short

    for rel in dict.fromkeys(NEEDED):
        out = dest / rel
        if out.exists() and out.stat().st_size > 0:
            print(f"  跳过 {rel}（已存在）")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        q = urllib.parse.urlencode({"Revision": "master", "FilePath": rel})
        url = f"{MODELSCOPE_API}/{repo}/repo?{q}"
        data = _get(url)
        out.write_bytes(data)
        print(f"  {rel:36s} {len(data):>12,d} B")

    return dest


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    p = download(repo)
    print(f"\n模型已就绪：{p}")
    # 校验能否被 sentence_transformers 加载
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(str(p), device="cpu")
    v = m.encode(["hello world"], normalize_embeddings=True)
    print(f"加载成功，维度 {m.get_sentence_embedding_dimension()}，样例向量范数 {float((v**2).sum() ** 0.5):.4f}")
