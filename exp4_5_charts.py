# -*- coding: utf-8 -*-
"""
生成实验四报告用图表（数据全部来自本实验真实运行结果）
  图1：检索策略调优对比（chunk_size × 查询指令前缀 → Hit@1 / Hit@3）
  图2：10 道测试题的 top1 相似度分布（含拒答阈值线，验证"库内/库外"可分）
  图5：生成层 LLM-as-Judge 评分——纯大模型 vs RAG（三维度 + 逐题分差）
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
FONT = None
for cand in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(cand):
        FONT = font_manager.FontProperties(fname=cand)
        break
if FONT:
    plt.rcParams["axes.unicode_minus"] = False


def fig1():
    data = json.load(open(os.path.join(BASE, "tune_result.json"), encoding="utf-8"))
    labels, h1, h3 = [], [], []
    for r in data:
        labels.append(f"chunk={r['chunk_size']}\n前缀={r['指令前缀']}")
        h1.append(int(r["Hit@1"].split("/")[0]))
        h3.append(int(r["Hit@3"].split("/")[0]))

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], h1, w, label="Hit@1（首条命中）", color="#4C78A8")
    b2 = ax.bar([i + w / 2 for i in x], h3, w, label="Hit@3（前三命中）", color="#F58518")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
                    f"{int(b.get_height())}/8", ha="center", fontsize=9)
    ax.set_ylabel("命中题数（共 8 题）", fontproperties=FONT)
    ax.set_title("检索策略调优：chunk_size 与查询指令前缀对命中率的影响", fontproperties=FONT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontproperties=FONT, fontsize=9)
    ax.set_ylim(0, 9)
    ax.legend(prop=FONT)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(BASE, "fig1_tune.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("图1 ->", p)


def fig2():
    # 拒答阈值须与 exp4_3_rag.SIM_THRESHOLD 保持一致
    SIM_THRESHOLD = 0.40
    rows = list(csv.DictReader(open(os.path.join(BASE, "eval_results.csv"), encoding="utf-8-sig")))
    names = [r["问题"][:18] + ("…" if len(r["问题"]) > 18 else "") for r in rows]
    sims = [float(r["Top1相似度"]) for r in rows]
    inkb = [r["库内"] == "是" for r in rows]
    colors = ["#4C78A8" if k else "#E45756" for k in inkb]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(range(len(rows)), sims, color=colors)
    ax.axvline(SIM_THRESHOLD, color="#54A24B", linestyle="--", linewidth=1.5)
    ax.text(SIM_THRESHOLD + 0.008, len(rows) - 0.35, f"拒答阈值 {SIM_THRESHOLD}", color="#54A24B",
            fontsize=9, fontproperties=FONT)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontproperties=FONT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Top1 余弦相似度", fontproperties=FONT)
    ax.set_title("10 道测试题的检索相似度分布（蓝=库内问题，红=库外问题）", fontproperties=FONT)
    for i, s in enumerate(sims):
        ax.text(s + 0.005, i, f"{s:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(sims) * 1.2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(BASE, "fig2_similarity.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("图2 ->", p)


def fig5():
    """图5：生成层评分对比（左：三维度均值；右：逐题 RAG − 纯大模型 分差）。"""
    s = json.load(open(os.path.join(BASE, "eval_summary.json"), encoding="utf-8"))
    jd = s["judge"]
    dims = [("accuracy", "准确性"), ("completeness", "完整性"), ("relevance", "相关性")]

    rows = list(csv.DictReader(open(os.path.join(BASE, "eval_results.csv"), encoding="utf-8-sig")))
    names = [r["问题"][:16] + ("…" if len(r["问题"]) > 16 else "") for r in rows]
    diff = [float(r["RAG均分"]) - float(r["纯大模型均分"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    x = range(len(dims))
    w = 0.36
    base_v = [jd["base"][d[0]] for d in dims]
    rag_v = [jd["rag"][d[0]] for d in dims]
    b1 = ax1.bar([i - w / 2 for i in x], base_v, w, label="纯大模型", color="#E45756")
    b2 = ax1.bar([i + w / 2 for i in x], rag_v, w, label="RAG", color="#4C78A8")
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
                     f"{b.get_height():.2f}", ha="center", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([d[1] for d in dims], fontproperties=FONT)
    ax1.set_ylim(0, 5.6)
    ax1.set_ylabel("LLM-as-Judge 评分（1-5 分）", fontproperties=FONT)
    ax1.set_title(f"三维度均值　综合 {jd['base']['overall']} → {jd['rag']['overall']}",
                  fontproperties=FONT)
    ax1.legend(prop=FONT)
    ax1.grid(axis="y", alpha=0.3)

    colors = ["#54A24B" if d >= 0 else "#E45756" for d in diff]
    ax2.barh(range(len(rows)), diff, color=colors)
    ax2.axvline(0, color="#333333", linewidth=1)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(names, fontproperties=FONT, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("RAG - 纯大模型（分）", fontproperties=FONT)
    ax2.set_title("逐题分差（绿=RAG 更好，红=纯大模型更好）", fontproperties=FONT)
    for i, d in enumerate(diff):
        # 用 ASCII 的 '+' / '-'，避免 SimHei 缺少 U+2212 导致方框
        lab = ("+" if d >= 0 else "-") + f"{abs(d):.2f}"
        ax2.text(d + (0.05 if d >= 0 else -0.05), i, lab,
                 va="center", ha="left" if d >= 0 else "right", fontsize=8)
    ax2.set_xlim(min(diff) - 0.6, max(diff) + 0.6)
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    p = os.path.join(BASE, "fig5_judge.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("图5 ->", p)


if __name__ == "__main__":
    fig1()
    fig2()
    fig5()
