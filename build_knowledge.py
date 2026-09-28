# -*- coding: utf-8 -*-
"""
实验4-1 前置：私有知识文档准备
--------------------------------------------------
从《大模型应用实训》实验指导书（.docx）中抽取正文，清洗后生成
纯文本知识库 knowledge.txt（本例约 2 万字，满足"不少于 5000 字"要求）。

为什么要清洗：
  1) 目录页是 Word 域代码（TOC / PAGEREF / HYPERLINK），不清洗会变成噪音 chunk；
  2) 页眉页脚、空行会打断段落边界，影响 RecursiveCharacterTextSplitter 的切分质量；
  3) 代码块是讲义的重要知识点，必须保留（后续问答会问到 API 调用写法）。
"""
import re
import zipfile
import html
import os

SRC_DOCX = r"D:\微信\xwechat_files\wxid_4pysdr5pa3r222_dac6\msg\file\2026-09\《大模型应用实训》实验指导书.docx"
OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.txt")


def extract_docx_text(path):
    """用标准库直接读 docx 的 document.xml，避免依赖 python-docx。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    # 段落结束 -> 换行；制表符/软回车还原
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


def clean(text):
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        # 丢弃 Word 目录域代码行
        if any(k in s for k in ("PAGEREF", "TOC \\o", "HYPERLINK")):
            continue
        # 丢弃封面的空下划线占位（学号/姓名等留空行）
        if re.fullmatch(r"[＿_—\-\s]{2,}", s):
            continue
        lines.append(s)
    body = "\n".join(lines)
    # 从正文起点"实训课程目标"开始，丢弃封面与目录
    start = body.find("实训课程目标")
    if start > 0:
        body = body[start:]
    return body.strip()


def main():
    raw = extract_docx_text(SRC_DOCX)
    text = clean(raw)

    # 封面上的课程元信息也是知识库的一部分（后续问答会问到学时、指导教师等）
    meta = (
        "课程基本信息\n"
        "课程名称：《大模型应用实训》\n"
        "指导教师：刘丽华\n"
        "学时：1学时\n"
        "教研室：信息与计算科学\n"
        "学院：数学与计算机科学学院\n"
        "编制时间：2026年9月\n\n"
    )

    # 注意：相邻字符串字面量会先拼接再参与 * 运算，
    # 写成 "...\n" "=" * 60 会把整句重复 60 遍，必须先用括号收口。
    header = (
        "《大模型应用实训》课程讲义知识库\n"
        "（本文件为实验四 RAG 实验的私有知识文档，内容摘自本课程实验指导书）\n"
    ) + "=" * 60 + "\n\n"
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(header + meta + text + "\n")

    n_chars = len(text) + len(meta)
    n_lines = text.count("\n") + 1
    print(f"知识文档已生成：{OUT_TXT}")
    print(f"字符数：{n_chars}（要求 ≥ 5000，{'达标' if n_chars >= 5000 else '不达标'}）")
    print(f"行数：{n_lines}")


if __name__ == "__main__":
    main()
