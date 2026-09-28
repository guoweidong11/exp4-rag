# -*- coding: utf-8 -*-
"""
实验4-1：文档加载与切分
--------------------------------------------------
要点：
  * RecursiveCharacterTextSplitter 按 ["\n\n", "\n", "。", "，", ""] 递归降级切分，
    优先保住段落与句子边界，避免把章节标题拦腰截断。
  * chunk_size=300 / chunk_overlap=50：重叠区让跨块语义不断裂，
    检索时即使命中边界也能拿到完整上下文。
  * 切分后落盘 chunks.json，供 4-2 向量化与 4-3 回取原文共用。
"""
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = os.path.dirname(os.path.abspath(__file__))
KB_TXT = os.path.join(BASE, "knowledge.txt")
CHUNKS_JSON = os.path.join(BASE, "chunks.json")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def main():
    with open(KB_TXT, encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 中文场景下显式补上中文句读边界，切分更贴合语义
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_text(text)

    print(f"共切分为 {len(chunks)} 个片段（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
    lens = [len(c) for c in chunks]
    print(f"片段长度：最小 {min(lens)} / 平均 {sum(lens)/len(lens):.1f} / 最大 {max(lens)}")

    # 抽查边界：确认章节标题未被截断、重叠区生效
    print("\n--- 抽查第 1、2、3 个片段 ---")
    for i in (0, 1, 2):
        print(f"\n[chunk {i}] ({len(chunks[i])} 字)\n{chunks[i][:280]}")

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"\n已保存：{CHUNKS_JSON}")


if __name__ == "__main__":
    main()
