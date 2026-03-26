from __future__ import annotations

import streamlit as st

from config import HF_MODEL_ID, HF_TOKEN, TOP_K_DEFAULT
from data_store import load_tables, retrieve
#from llm_client import call_deepseek_r1
from llm_client import call_llama_via_router


st.set_page_config(
    page_title="DeepSeek-R1 RAG Demo (Modular)",
    page_icon="📊",
    layout="wide",
)

st.title("📊 DeepSeek-R1 RAG Demo - Campaign Analytics (Modular)")
st.markdown(
    """
This is a lightweight modular RAG (Retrieval-Augmented Generation) demo using:

- **DeepSeek-R1** via Hugging Face Inference as the reasoning LLM
- **all-MiniLM-L6-v2** sentence embeddings for semantic search
- 100-row synthetic tables for **conversions** and **channel engagement**

Everything is split into small modules so you can reuse pieces easily:
- `config.py` – configuration & paths
- `data_store.py` – loading data + vector retrieval
- `llm_client.py` – Hugging Face DeepSeek-R1 client
- `app.py` – Streamlit UI
"""
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("### RAG Parameters")
    top_k = st.slider("Top-K retrieved rows", min_value=1, max_value=20, value=TOP_K_DEFAULT)

    st.markdown("### DeepSeek-R1 Settings")
    st.caption("Model and token can be overridden via environment variables.")
    st.text_input("HF Model ID", value=HF_MODEL_ID, disabled=True)
    token_set = bool(HF_TOKEN)
    st.write("HF Token set:", "✅" if token_set else "⚠️ Not set (public models only)")

    st.markdown("### Data Preview")
    conv_df, eng_df = load_tables()
    with st.expander("Conversion Table (first 20 rows)", expanded=False):
        st.dataframe(conv_df.head(20), use_container_width=True)
    with st.expander("Channel Engagement Table (first 20 rows)", expanded=False):
        st.dataframe(eng_df.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Tip:** For deployment on Streamlit Cloud, set `HF_TOKEN` as a secret "
        "and keep using `deepseek-ai/DeepSeek-R1` (or another compatible DeepSeek model)."
    )

# Chat-style interface
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("## 💬 Ask a Question")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask something about campaign performance, treatments, or channels...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and asking DeepSeek-R1..."):
            context, idxs = retrieve(user_query, top_k=top_k)
            #answer = call_deepseek_r1(user_query, context)
            answer = call_llama_via_router(user_query, context)


            st.markdown(answer)

            with st.expander("🔍 Debug: Retrieved Context"):
                st.text(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption(
    "Modular demo app: DeepSeek-R1 + MiniLM RAG over 100-row synthetic conversions & "
    "channel engagement tables."
)
