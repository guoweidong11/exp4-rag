# -*- coding: utf-8 -*-
"""
实验四 4.4 任务：10 个测试问题 —— 纯大模型 vs RAG 对比评估
--------------------------------------------------
评估分两层：
  1) 检索层（离线可测）：人工标注每个问题的"黄金片段关键词"，统计 Hit@1 / Hit@3，
     即 top-k 检索结果里是否包含真正能回答该问题的文本块。
  2) 生成层（需 API Key）：同一问题分别用
       - 纯大模型（不带任何资料）作答
       - RAG（带检索资料）作答
     再用"大模型当裁判（LLM-as-Judge）"从准确性/完整性/相关性三维度 1-5 分打分。

无 API Key 时自动只跑第 1 层，生成层留空，配置密钥后重跑即可补全。
输出：eval_results.csv + eval_results.md + eval_summary.json
"""
import csv
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from exp4_3_rag import (EMBED_MODEL, LLM_MODEL, LLM_NAME, SIM_THRESHOLD,  # noqa: E402
                        chat_json, chat_text, get_kb)

BASE = os.path.dirname(os.path.abspath(__file__))

# q=问题  ref=参考答案  key=黄金片段关键词（用于判定检索是否命中）
# in_kb=False 表示知识库中【没有】答案，正确行为是如实说明资料不足
TESTS = [
    {"q": "本课程的学时是多少？", "ref": "1学时", "key": "1学时", "in_kb": True},
    {"q": "本课程的指导教师是谁？", "ref": "刘丽华", "key": "刘丽华", "in_kb": True},
    {"q": "本课程由哪个学院、哪个教研室开设？",
     "ref": "数学与计算机科学学院，信息与计算科学教研室", "key": "信息与计算科学", "in_kb": True},
    {"q": "实验四要求准备的私有知识文档至少多少字？", "ref": "不少于5000字", "key": "5000字", "in_kb": True},
    {"q": "实验4-2 使用的 Embedding 模型叫什么？", "ref": "BAAI/bge-small-zh-v1.5",
     "key": "bge-small-zh", "in_kb": True},
    {"q": "文档切分时 chunk_size 和 chunk_overlap 分别是多少？",
     "ref": "chunk_size=300，chunk_overlap=50", "key": "chunk_overlap=50", "in_kb": True},
    {"q": "实验4-3 检索增强问答中，调用大模型时 temperature 设为多少？",
     "ref": "0.1（以较低 temperature 调用）", "key": "temperature（0.1）", "in_kb": True},
    {"q": "实验二的猜谜游戏里，出题和裁判两个环节的温度分别设成多少？",
     "ref": "出题 temperature=1.0，裁判 temperature=0.0", "key": "高温（1.0）", "in_kb": True},
    {"q": "2026 年诺贝尔物理学奖得主是谁？",
     "ref": "知识库中没有该信息，应如实说明资料不足", "key": None, "in_kb": False},
    {"q": "学校三号食堂本周一的菜单是什么？",
     "ref": "知识库中没有该信息，应如实说明资料不足", "key": None, "in_kb": False},
]

JUDGE_PROMPT = """你是客观严谨的评分裁判。请对比参考答案，对待评答案打分。

【问题】{question}
【参考答案】{reference}
【待评答案】{answer}

评分维度（各 1-5 分）：
- accuracy 准确性：事实是否与参考答案一致，是否存在编造
- completeness 完整性：是否覆盖了参考答案的关键点
- relevance 相关性：是否紧扣问题、无无关内容

仅返回JSON：{{"accuracy": 1-5, "completeness": 1-5, "relevance": 1-5, "reason": "一句话理由"}}"""


def pure_llm_answer(client, question):
    """纯大模型基线：不带任何资料，完全依赖参数内知识。"""
    return chat_text(client, [
        {"role": "user", "content":
         f"请回答：{question}\n要求150字以内，若不确定请说明。"}
    ], temperature=0.1, max_tokens=400)


def llm_judge(client, question, answer, reference):
    return chat_json(client, [
        {"role": "user", "content":
         JUDGE_PROMPT.format(question=question, answer=answer, reference=reference)}
    ], temperature=0.0)


def main():
    kb = get_kb()
    has_llm = kb.client is not None
    print(f"检索层评估：共 {len(TESTS)} 题｜生成层评估："
          f"{'开启（' + LLM_NAME + '/' + LLM_MODEL + '）' if has_llm else '未开启（无 API Key）'}")

    rows = []
    hit1 = hit3 = 0
    gate_ok_n = 0
    dims = {"base": {"accuracy": [], "completeness": [], "relevance": []},
            "rag": {"accuracy": [], "completeness": [], "relevance": []}}
    for t in TESTS:
        hits = kb.retrieve(t["q"], top_k=3)
        texts = [c for c, _ in hits]
        if t["in_kb"]:
            h1 = t["key"] in texts[0]
            h3 = any(t["key"] in c for c in texts)
            hit1 += int(h1)
            hit3 += int(h3)
        else:
            h1 = h3 = None  # 库外问题不考察命中率，考察是否拒答

        # 拒答线判定：top1 相似度 < 阈值 -> 视为"资料中没有"，应拒答
        s1 = hits[0][1] if hits else 0.0
        if t["in_kb"]:
            gate_ok = s1 >= SIM_THRESHOLD
            gate = "✓ 正常作答" if gate_ok else "✗ 误拒答"
        else:
            gate_ok = s1 < SIM_THRESHOLD
            gate = "✓ 正确拒答" if gate_ok else "✗ 未拒答"
        gate_ok_n += int(gate_ok)

        row = {
            "问题": t["q"],
            "库内": "是" if t["in_kb"] else "否",
            "参考答案": t["ref"],
            "Top1相似度": round(hits[0][1], 4) if hits else "",
            "阈值判定": gate,
            "Hit@1": "" if h1 is None else ("✓" if h1 else "✗"),
            "Hit@3": "" if h3 is None else ("✓" if h3 else "✗"),
        }

        if has_llm:
            # 单题调用失败不影响整体评估（网络抖动 / 限流）
            try:
                base = pure_llm_answer(kb.client, t["q"])
                rag_ans, _ = kb.answer(t["q"], top_k=3)
                jb = llm_judge(kb.client, t["q"], base, t["ref"])
                jr = llm_judge(kb.client, t["q"], rag_ans, t["ref"])
            except Exception as e:
                print(f"     [生成层跳过] {type(e).__name__}: {str(e)[:80]}")
                base = rag_ans = None
                jb = jr = {}
            if jb and jr:
                for d in ("accuracy", "completeness", "relevance"):
                    dims["base"][d].append(float(jb.get(d, 0)))
                    dims["rag"][d].append(float(jr.get(d, 0)))
                bs = sum(dims["base"][d][-1] for d in dims["base"]) / 3
                rs = sum(dims["rag"][d][-1] for d in dims["rag"]) / 3
                row.update({
                    "纯大模型回答": base,
                    "RAG回答": rag_ans,
                    "纯大模型均分": round(bs, 2),
                    "RAG均分": round(rs, 2),
                    "裁判理由(RAG)": jr.get("reason", ""),
                })
                print(f"     纯大模型 {bs:.2f} vs RAG {rs:.2f}")
        rows.append(row)
        print(f"  · {t['q'][:28]} -> Hit@1={row['Hit@1'] or '-'} Hit@3={row['Hit@3'] or '-'}")

    n_in = sum(1 for t in TESTS if t["in_kb"])
    summary = {
        "embed_model": EMBED_MODEL,
        "llm_model": f"{LLM_NAME}/{LLM_MODEL}" if has_llm else None,
        "sim_threshold": SIM_THRESHOLD,
        "hit@1": f"{hit1}/{n_in}",
        "hit@3": f"{hit3}/{n_in}",
        "gate_correct": f"{gate_ok_n}/{len(TESTS)}",
    }
    if has_llm:
        summary["judge"] = {
            k: {d: round(sum(v[d]) / len(v[d]), 2) for d in v}
            for k, v in dims.items()
        }
        n_judged = len(dims["base"]["accuracy"])
        summary["judge"]["base"]["overall"] = round(
            sum(sum(v) for v in dims["base"].values()) / (3 * n_judged), 2)
        summary["judge"]["rag"]["overall"] = round(
            sum(sum(v) for v in dims["rag"].values()) / (3 * n_judged), 2)
        print("\n生成层（LLM-as-Judge，1-5 分）：")
        for k, label in (("base", "纯大模型"), ("rag", "RAG  ")):
            s = summary["judge"][k]
            print(f"  {label}：准确性 {s['accuracy']}｜完整性 {s['completeness']}"
                  f"｜相关性 {s['relevance']}｜综合 {s['overall']}")

    print(f"\n检索命中率：Hit@1 = {hit1}/{n_in} = {hit1/n_in:.0%}｜"
          f"Hit@3 = {hit3}/{n_in} = {hit3/n_in:.0%}")
    print(f"拒答线（阈值 {SIM_THRESHOLD}）判定正确：{gate_ok_n}/{len(TESTS)} = "
          f"{gate_ok_n/len(TESTS):.0%}")

    csv_path = os.path.join(BASE, "eval_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(BASE, "eval_results.md")
    cols = [c for c in rows[0].keys() if c not in ("参考答案",)]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            cells = [str(r[c]).replace("\n", "<br>")[:200] for c in cols]
            f.write("| " + " | ".join(cells) + " |\n")

    summary_path = os.path.join(BASE, "eval_summary.json")
    json.dump(summary, open(summary_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"已保存：{csv_path}\n已保存：{md_path}\n已保存：{summary_path}")


if __name__ == "__main__":
    main()
