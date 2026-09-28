# -*- coding: utf-8 -*-
"""
检索策略调优实验（对应实验四"结果分析"章节）
--------------------------------------------------
对照两组变量，用 8 道有唯一正确答案的问题量化检索质量：
  变量A：查询是否加 bge 官方指令前缀
  变量B：chunk_size = 300（指导书推荐值） vs 500

指标：Hit@1（首条命中）/ Hit@3（前三命中） / 平均相似度
"""
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
SEPS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

CASES = [
    ("本课程的学时是多少？", "1学时"),
    ("本课程的指导教师是谁？", "刘丽华"),
    ("本课程由哪个学院、哪个教研室开设？", "信息与计算科学"),
    ("实验四要求准备的私有知识文档至少多少字？", "5000字"),
    ("实验4-2 使用的 Embedding 模型叫什么？", "bge-small-zh"),
    ("文档切分时 chunk_size 和 chunk_overlap 分别是多少？", "chunk_overlap=50"),
    ("实验4-3 检索增强问答中，调用大模型时 temperature 设为多少？", "temperature（0.1）"),
    ("实验二的猜谜游戏里，出题和裁判两个环节的温度分别设成多少？", "高温（1.0）"),
]


def evaluate(model, index, chunks, use_instruction):
    hit1 = hit3 = 0
    sims = []
    miss = []
    for q, key in CASES:
        qq = QUERY_INSTRUCTION + q if use_instruction else q
        qv = model.encode([qq], normalize_embeddings=True).astype(np.float32)
        s, i = index.search(qv, 3)
        texts = [chunks[j] for j in i[0]]
        h1, h3 = key in texts[0], any(key in c for c in texts)
        hit1 += int(h1)
        hit3 += int(h3)
        sims.append(float(s[0][0]))
        if not h3:
            miss.append(q)
    return hit1, hit3, float(np.mean(sims)), miss


def main():
    text = open(os.path.join(BASE, "knowledge.txt"), encoding="utf-8").read()
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    report = []

    for cs in (300, 500):
        ov = 50 if cs == 300 else 80
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs, chunk_overlap=ov, separators=SEPS)
        chunks = splitter.split_text(text)
        t0 = time.time()
        emb = model.encode(chunks, normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False).astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        for use_ins in (False, True):
            h1, h3, sim, miss = evaluate(model, index, chunks, use_ins)
            row = {"chunk_size": cs, "overlap": ov, "块数": len(chunks),
                   "指令前缀": "是" if use_ins else "否",
                   "Hit@1": f"{h1}/8", "Hit@3": f"{h3}/8",
                   "平均相似度": round(sim, 4), "耗时s": round(time.time() - t0, 1),
                   "未命中": miss}
            report.append(row)
            print(f"chunk={cs} 前缀={row['指令前缀']} -> Hit@1 {h1}/8, Hit@3 {h3}/8, "
                  f"平均相似度 {sim:.4f}, 块数 {len(chunks)}")
            if miss:
                print(f"   未命中：{miss}")

    with open(os.path.join(BASE, "tune_result.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n调优结果已保存 tune_result.json")


if __name__ == "__main__":
    main()
