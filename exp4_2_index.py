# -*- coding: utf-8 -*-
"""
实验4-2：向量化与 FAISS 索引
--------------------------------------------------
要点：
  * Embedding 模型 BAAI/bge-small-zh-v1.5（约 100MB，中文语义检索效果好）。
    国内网络走 hf-mirror 镜像，脚本里已自动设置 HF_ENDPOINT。
  * normalize_embeddings=True：向量模长为 1，此时「内积 == 余弦相似度」，
    所以可以直接用 faiss.IndexFlatIP（精确内积索引），无需训练、结果精确。
  * chunks[i] 与索引中第 i 条向量一一对应，检索回的 ids 直接回取原文。
  * 本步骤完全本地运行，不消耗任何 API 费用。
"""
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS_JSON = os.path.join(BASE, "chunks.json")
INDEX_FILE = os.path.join(BASE, "kb.index")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"载入 {len(chunks)} 个文本块")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,      # 关键：模长归一，内积即余弦
        batch_size=32,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"向量化完成，耗时 {time.time()-t0:.1f}s，向量形状 {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)      # 精确内积索引（百万级以内无需 IVF/HNSW）
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS 索引已建立：{index.ntotal} 条向量，维度 {dim}")

    # ---- 正确性验证：第 3 个块与自身做点积，归一化后应 ≈ 1.0 ----
    v3 = embeddings[2:3]
    scores, ids = index.search(v3, 3)
    print(f"\n自检：chunk[2] 自检索 top3 -> ids={ids[0].tolist()} 相似度={[round(float(s),4) for s in scores[0]]}")
    assert abs(scores[0][0] - 1.0) < 1e-3, "自检失败：向量未正确归一化"
    print("自检通过：自身相似度为 1.0，说明归一化与索引建立正确。")

    # ---- 语义检索演示（证明是语义匹配而非关键词匹配）----
    demo = [
        "这门课一共安排了多少学时？",
        "做 RAG 用什么模型把文本变成向量？",
        "如果玩家输入恶意提示词该怎么防？",
    ]
    print("\n--- 语义检索演示（top_k=2，查询加 bge 官方指令前缀）---")
    for q in demo:
        qv = model.encode(["为这个句子生成表示以用于检索相关文章：" + q],
                          normalize_embeddings=True).astype(np.float32)
        s, i = index.search(qv, 2)
        print(f"\n问：{q}")
        for rank, (cid, sc) in enumerate(zip(i[0], s[0]), 1):
            print(f"  [{rank}] 相似度 {float(sc):.4f} | {chunks[cid][:90]}...")

    print(f"\n索引文件：{INDEX_FILE}")


if __name__ == "__main__":
    main()
