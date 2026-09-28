# -*- coding: utf-8 -*-
"""
实验4-3：检索增强问答（RAG）
--------------------------------------------------
链路：问题向量化 -> FAISS 检索 top_k -> 拼接 context -> 大模型有据作答 -> 返回引用来源

关键设计：
  1) 检索与生成必须用【同一个】Embedding 模型，否则向量空间不一致，检索失效。
  2) 提示词里那句"请仅根据以下资料回答，若资料不足请如实说明"是防幻觉的核心：
     前半句把模型 grounding 在资料上，后半句给模型一个"可以说不知道"的出口。
  3) temperature=0.1：问答追求稳定与忠实，不需要创造性。
  4) 无 API Key 时自动降级为"引用式作答"（直接回取最相关片段），
     保证离线也能完整演示检索链路；配置密钥后即自动切换为真实生成。

密钥配置（三选一，脚本会自动探测）：
  a) 环境变量：ZHIPU_API_KEY / DEEPSEEK_API_KEY / LLM_API_KEY
  b) 本目录 .env 文件：ZHIPU_API_KEY=xxx（勿提交到公开仓库）
  本实验实测使用 智谱 GLM（glm-4-flash，OpenAI 兼容接口）。
"""
import json
import os


def _pick_hf_endpoint(timeout=4):
    """自动选择 HuggingFace 端点：能直连官方就用官方，否则走国内镜像 hf-mirror。
    本机（国内）走镜像，Streamlit Cloud（海外）直连官方，两端都无需手改配置。
    若已显式设置 HF_ENDPOINT 环境变量，则尊重用户设置。"""
    if os.environ.get("HF_ENDPOINT"):
        return
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    except OSError:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


_pick_hf_endpoint()

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
KB_TXT = os.path.join(BASE, "knowledge.txt")
CHUNKS_JSON = os.path.join(BASE, "chunks.json")
INDEX_FILE = os.path.join(BASE, "kb.index")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# 与 exp4_1_split.py 保持一致的切分参数（自动重建索引时使用）
CHUNK_SIZE, CHUNK_OVERLAP = 300, 50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
# bge 系列官方建议：短查询检索时加该指令前缀（文档侧不加）。
# 实测（见 exp4_tune.py）：加前缀后 Hit@1 由 6/8 提升到 7/8。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
# 相似度阈值（拒答线）：10 题实测——库内 8 题 top1 相似度最低 0.4318，
# 库外 2 题最高 0.3818，两类完全可分，故取两者中间值 0.40 作为拒答线。
SIM_THRESHOLD = 0.40

# ---------------- 大模型供应商自动探测 ----------------
# 三家都走 OpenAI 兼容协议，只需换 base_url + model，业务代码无需改动。
# 这也是 RAG 工程里的常见做法：把供应商差异收敛到配置层。
PROVIDERS = [
    # (环境变量名, base_url, 模型名, 展示名)
    ("ZHIPU_API_KEY", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", "智谱GLM"),
    ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat", "DeepSeek"),
    ("LLM_API_KEY", "https://api.deepseek.com", "deepseek-chat", "DeepSeek"),
]
LLM_MODEL, LLM_BASE_URL, LLM_NAME = "glm-4-flash", "https://open.bigmodel.cn/api/paas/v4", "智谱GLM"


def _load_env_file():
    """读取本目录 .env（KEY=VALUE 格式），不覆盖已有环境变量。"""
    path = os.path.join(BASE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _load_streamlit_secrets():
    """Streamlit Cloud：从 st.secrets 读取密钥注入环境变量。
    云端没有 .env 文件，密钥配在 App Settings → Secrets 里；本地未配置时静默跳过。"""
    try:
        import streamlit as st
        for k in ("ZHIPU_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY"):
            v = st.secrets.get(k)
            if v:
                os.environ.setdefault(k, str(v))
    except Exception:
        pass


_load_env_file()
_load_streamlit_secrets()

PROMPT_TEMPLATE = """请仅根据以下资料回答问题，若资料不足请如实说明。

【资料】
{context}

【问题】{question}

要求：答案控制在 150 字以内，忠于资料，不要编造资料中没有的信息。"""


def chat_text(client, messages, temperature=0.1, max_tokens=400):
    """统一的对话调用（屏蔽不同供应商的细微差异）。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_json(client, messages, temperature=0.0):
    """要求模型返回 JSON；若供应商不支持 response_format 则自动降级为文本解析。"""
    import re
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        raw = chat_text(client, messages, temperature=temperature, max_tokens=600)
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


class RAGKB:
    """知识库单例：加载 chunks + FAISS 索引 + Embedding 模型。"""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self._ensure_index()      # 索引缺失时现场重建（换机器 / 首次部署兜底）
        with open(CHUNKS_JSON, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.index = faiss.read_index(INDEX_FILE)
        self.client = self._build_client()

    def _ensure_index(self):
        """chunks.json 或 kb.index 缺失时，用 knowledge.txt 现场切分并建索引。
        这样仓库里只需提交 knowledge.txt 也能跑起来（Streamlit Cloud 首次启动会多花约 30 秒）。"""
        if os.path.exists(CHUNKS_JSON) and os.path.exists(INDEX_FILE):
            return
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text = open(KB_TXT, encoding="utf-8").read()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS).split_text(text)
        with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=1)
        emb = np.asarray(self.model.encode(chunks, normalize_embeddings=True,
                                           batch_size=32), dtype=np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, INDEX_FILE)
        print(f"[自动构建] 已生成 {len(chunks)} 个文本块与索引 {INDEX_FILE}")

    @staticmethod
    def _build_client():
        """按 PROVIDERS 顺序探测可用密钥；都读不到就返回 None（离线模式）。"""
        global LLM_MODEL, LLM_BASE_URL, LLM_NAME
        for env_key, base_url, model, name in PROVIDERS:
            api_key = os.environ.get(env_key)
            if not api_key:
                continue
            try:
                from openai import OpenAI
                LLM_MODEL, LLM_BASE_URL, LLM_NAME = model, base_url, name
                return OpenAI(api_key=api_key, base_url=base_url)
            except Exception:
                return None
        return None

    def retrieve(self, question, top_k=3):
        """返回 [(chunk, score), ...]，按相似度降序。"""
        q = QUERY_INSTRUCTION + question          # 查询侧加指令前缀，文档侧不加
        qv = self.model.encode([q], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)
        scores, ids = self.index.search(qv, top_k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def answer(self, question, top_k=3):
        """返回 (answer, sources)；sources 为 [(文本, 相似度), ...]"""
        hits = self.retrieve(question, top_k)
        if not hits:
            return "知识库中没有检索到相关内容。", []

        context = "\n\n".join(c for c, _ in hits)

        if self.client is None:
            # 离线降级：直接引用最相关片段（等价于"纯检索"基线）
            if hits[0][1] < SIM_THRESHOLD:
                return (f"[离线引用模式] 资料中没有与该问题相关的内容"
                        f"（最高相似度 {hits[0][1]:.4f} < 阈值 {SIM_THRESHOLD}），"
                        f"无法回答，请换个问题或补充资料。"), hits
            best = hits[0][0].strip()
            answer = ("[离线引用模式 · 未配置 API Key]\n"
                      f"资料中最相关的一段如下：\n{best}")
            return answer, hits

        # 相似度低于阈值 -> 直接拒答，不把资料喂给模型（避免模型硬凑答案产生幻觉）
        if hits[0][1] < SIM_THRESHOLD:
            return (f"资料中没有与该问题相关的内容（最高相似度 "
                    f"{hits[0][1]:.4f} < 阈值 {SIM_THRESHOLD}），无法回答，"
                    f"请换个问题或补充资料。"), hits

        answer = chat_text(self.client, [
            {"role": "user", "content":
             PROMPT_TEMPLATE.format(context=context, question=question)}
        ], temperature=0.1, max_tokens=400)
        return answer, hits


_KB = None


def get_kb():
    """全局复用一个知识库实例（Streamlit 下避免每次交互重复加载 100MB 模型）。"""
    global _KB
    if _KB is None:
        _KB = RAGKB()
    return _KB


def rag_answer(question, top_k=3):
    """对外统一入口：返回 (answer, sources)。"""
    return get_kb().answer(question, top_k)


if __name__ == "__main__":
    kb = get_kb()
    print(f"知识库就绪：{len(kb.chunks)} 个文本块，"
          f"生成模式={'在线(' + LLM_NAME + '/' + LLM_MODEL + ')' if kb.client else '离线引用'}")

    questions = [
        "本课程的学时是多少？指导教师是谁？",          # 库内问题：应准确回答
        "实验四要求准备的私有知识文档至少多少字？",      # 库内问题
        "向量数据库除了 FAISS 还有哪些？",             # 库内（思考题）
        "2026 年世界杯冠军是哪支球队？",               # 库外问题：应如实说明资料不足
    ]
    for q in questions:
        ans, src = kb.answer(q, top_k=3)
        print("\n" + "=" * 60)
        print(f"问：{q}")
        print(f"答：{ans}")
        print("引用来源：")
        for i, (c, s) in enumerate(src, 1):
            print(f"  [{i}] 相似度 {s:.4f} | {c[:100]}...")
