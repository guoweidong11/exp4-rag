# -*- coding: utf-8 -*-
"""
实验4-4：Streamlit 问答 APP
--------------------------------------------------
运行：streamlit run app.py

要点：
  * Streamlit 每次交互都会【重新执行整个脚本】，模型与索引必须用
    @st.cache_resource 缓存，否则每次提问都要重新加载 100MB 模型。
  * 对话历史存在 st.session_state.messages，这是唯一能跨交互保留的地方。
  * st.expander 折叠引用来源，方便用户核查答案依据（RAG 的可解释性）。
"""
import os  # noqa: E402

import streamlit as st  # noqa: E402

# 注意：不要在这里写死 HF_ENDPOINT。
# exp4_3_rag 导入时会自动探测端点（国内走 hf-mirror，海外/Streamlit Cloud 直连官方），
# 同时会把 Streamlit Secrets 里的 API Key 注入环境变量。
from exp4_3_rag import get_kb  # noqa: E402

st.set_page_config(page_title="课程讲义问答助手", page_icon="📚")
st.title("📚 课程讲义问答助手（RAG）")
st.caption("基于《大模型应用实训》实验指导书构建的检索增强问答系统")


@st.cache_resource(show_spinner="正在加载知识库与向量模型…")
def load_kb():
    return get_kb()


kb = load_kb()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("知识库状态")
    st.metric("文本块数量", len(kb.chunks))
    st.metric("索引向量数", kb.index.ntotal)
    st.caption(f"嵌入后端：{kb.model.backend}　|　索引：{os.path.basename(kb.index_file)}")
    from exp4_3_rag import LLM_MODEL, LLM_NAME
    mode = f"在线生成（{LLM_NAME}/{LLM_MODEL}）" if kb.client else "离线引用模式"
    st.info(f"当前模式：{mode}")
    if not kb.client:
        st.caption("在本目录 .env 里配置 ZHIPU_API_KEY（或设置环境变量）后重启，"
                   "即切换为大模型生成作答。")

    top_k = st.slider("检索条数 top_k", 1, 5, 3,
                      help="过小可能检索不全，过大会引入无关内容并增加 token 开销")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- 对话历史 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("查看引用来源"):
                for i, (c, s) in enumerate(m["sources"], 1):
                    st.caption(f"[{i}] 相似度 {s:.4f}\n\n{c}")

# ---------- 输入与应答 ----------
def ask(question):
    """一轮问答：渲染用户气泡 + 助手气泡（含引用来源折叠面板）。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            answer, sources = kb.answer(question, top_k=top_k)
        st.markdown(answer)
        if sources:
            with st.expander("查看引用来源"):
                for i, (c, s) in enumerate(sources, 1):
                    st.caption(f"[{i}] 相似度 {s:.4f}\n\n{c}")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if question := st.chat_input("请输入你的问题（例如：实验四的思考题有哪些？）"):
    ask(question)

# 支持 URL 直接提问（便于分享具体问题、也便于自动化截图）：
#   http://localhost:8501/?q=实验四要求准备的知识文档至少多少字
prefill = st.query_params.get("q")
if prefill and not st.session_state.get("_prefill_done"):
    st.session_state["_prefill_done"] = True
    ask(prefill)
