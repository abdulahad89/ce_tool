"""
Campaign Analytics RAG Agent — Streamlit App
Deployable on Streamlit Cloud with OpenAI or Google Gemini.
"""

import streamlit as st
import pandas as pd
import io
import time
from sample_data import get_sample_dataframes

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Campaign RAG Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f3460; }
    .metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

    /* Chat bubbles */
    .user-bubble {
        background: #0f3460;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .assistant-bubble {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 90%;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Context expander */
    .context-doc {
        background: #fefce8;
        border-left: 3px solid #eab308;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #713f12;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ready   { background: #dcfce7; color: #166534; }
    .status-pending { background: #fef9c3; color: #713f12; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background: #f8fafc; }

    /* Remove Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ─────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Agent Configuration")

    provider = st.selectbox(
        "🤖 Model Provider",
        ["OpenAI", "Gemini"],
        help="Choose the LLM provider for embeddings and generation.",
    )

    if provider == "OpenAI":
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your OpenAI API key. Never shared or stored.",
        )
        model_choice = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o-mini is fast and cost-effective for demos.",
        )
    else:
        api_key = st.text_input(
            "🔑 Google API Key",
            type="password",
            placeholder="AIza...",
            help="Your Google AI Studio API key.",
        )
        model_choice = st.selectbox(
            "Model",
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="gemini-2.0-flash is fastest for demos.",
        )

    st.markdown("---")
    top_k = st.slider(
        "📄 Retrieved Documents (Top-K)",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of relevant data chunks retrieved per query.",
    )

    st.markdown("---")

    init_btn = st.button(
        "🚀 Initialize RAG Agent",
        use_container_width=True,
        type="primary",
        disabled=not api_key,
    )

    if not api_key:
        st.warning("⚠️ Enter your API key above to begin.")

    # Status
    st.markdown("---")
    st.markdown("**Agent Status**")
    if st.session_state.initialized:
        st.markdown('<span class="status-badge status-ready">✅ Ready</span>', unsafe_allow_html=True)
        st.caption(f"Provider: {provider} | Model: {model_choice}")
        st.caption(f"Vector DB: ChromaDB (in-memory)")
    else:
        st.markdown('<span class="status-badge status-pending">⏳ Not Initialized</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.initialized:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **📦 Data Sources**
    - `Conversion Data` — 100 rows  
      campaigns × treatments × segments
    - `Engagement Data` — 100 rows  
      channels × campaigns
    - `Summaries` — 5 per-campaign  
      aggregated fact sheets
    """)


# ─────────────────────────────────────────────────────────────
# Initialization handler
# ─────────────────────────────────────────────────────────────
if init_btn and api_key:
    with st.spinner("🔧 Building vector index — embedding all campaign data…"):
        try:
            from rag_engine import RAGEngine
            st.session_state.rag_engine = RAGEngine(
                api_key=api_key,
                provider=provider,
                model=model_choice,
                top_k=top_k,
            )
            st.session_state.initialized = True
            st.session_state.chat_history = []
            st.success("✅ RAG Agent is ready! Ask your first question below.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Campaign Analytics RAG Agent</h1>
    <p>Ask natural-language questions about campaign conversions, channel engagement, A/B treatment performance, and more.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_chat, tab_data, tab_about = st.tabs(["💬 Chat Agent", "📋 Data Explorer", "ℹ️ About"])


# ═══════════════════════════════════════════
# TAB 1 — Chat Agent
# ═══════════════════════════════════════════
with tab_chat:
    if not st.session_state.initialized:
        st.info("👈 Configure your API key and click **Initialize RAG Agent** in the sidebar to get started.")

        st.markdown("### 💡 Example Questions You Can Ask")
        examples = [
            "Which campaign has the highest overall conversion rate?",
            "Compare Control vs Optimized treatment performance across all campaigns.",
            "Which channel has the best engagement score for Summer Launch?",
            "What is the cost per conversion for Holiday Promo's Variant A treatment?",
            "Which segment (New Users, Existing Users, High Value, All Users) converts best?",
            "Compare total spend vs engagement score across all channels for Retention Boost.",
            "What is the average CTR for Email channel across all campaigns?",
            "Which campaign had the highest number of conversions for the High Value segment?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            cols[i % 2].markdown(f"- *{ex}*")

    else:
        # ── Render chat history ──────────────────────────────
        for turn in st.session_state.chat_history:
            st.markdown(
                f'<div class="user-bubble">🧑 {turn["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="assistant-bubble">🤖 {turn["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if turn.get("docs"):
                with st.expander(f"📄 Retrieved Context ({len(turn['docs'])} chunks)", expanded=False):
                    for i, doc in enumerate(turn["docs"]):
                        st.markdown(
                            f'<div class="context-doc"><b>Doc {i+1}:</b> {doc}</div>',
                            unsafe_allow_html=True,
                        )

        # ── Suggested questions ──────────────────────────────
        if not st.session_state.chat_history:
            st.markdown("#### 💡 Try asking:")
            suggestions = [
                "Which campaign has the highest conversion rate?",
                "Compare treatment variants for Holiday Promo",
                "Which channel drives the best engagement score?",
                "What is the average CTR for Email across campaigns?",
            ]
            s_cols = st.columns(2)
            for i, s in enumerate(suggestions):
                if s_cols[i % 2].button(s, key=f"sugg_{i}"):
                    st.session_state._prefill = s
                    st.rerun()

        # ── Chat input ───────────────────────────────────────
        prefill = st.session_state.pop("_prefill", "")
        user_input = st.chat_input(
            "Ask about campaign performance, conversions, engagement…",
            key="chat_input",
        )

        # Handle prefill from suggestion buttons
        query = prefill or user_input

        if query:
            with st.spinner("🔍 Retrieving context & generating answer…"):
                try:
                    answer, docs = st.session_state.rag_engine.query(query)
                    st.session_state.chat_history.append(
                        {"question": query, "answer": answer, "docs": docs}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")


# ═══════════════════════════════════════════
# TAB 2 — Data Explorer
# ═══════════════════════════════════════════
with tab_data:
    df_conv, df_eng = get_sample_dataframes()

    st.markdown("### 📊 Dataset Overview")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""
    <div class="metric-card">
        <div class="value">{len(df_conv)}</div>
        <div class="label">Conversion Records</div>
    </div>""", unsafe_allow_html=True)
    m2.markdown(f"""
    <div class="metric-card">
        <div class="value">{len(df_eng)}</div>
        <div class="label">Engagement Records</div>
    </div>""", unsafe_allow_html=True)
    m3.markdown(f"""
    <div class="metric-card">
        <div class="value">{df_conv['campaign_id'].nunique()}</div>
        <div class="label">Unique Campaigns</div>
    </div>""", unsafe_allow_html=True)
    m4.markdown(f"""
    <div class="metric-card">
        <div class="value">{df_eng['channel'].nunique()}</div>
        <div class="label">Unique Channels</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    sub1, sub2 = st.tabs(["📈 Conversion Data", "📡 Channel Engagement Data"])

    with sub1:
        st.markdown("#### Campaign Conversion Data")

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        camp_filter = fc1.multiselect(
            "Campaign", df_conv["campaign_name"].unique(), default=list(df_conv["campaign_name"].unique())
        )
        treat_filter = fc2.multiselect(
            "Treatment", df_conv["treatment"].unique(), default=list(df_conv["treatment"].unique())
        )
        seg_filter = fc3.multiselect(
            "Segment", df_conv["segment"].unique(), default=list(df_conv["segment"].unique())
        )

        filtered = df_conv[
            df_conv["campaign_name"].isin(camp_filter)
            & df_conv["treatment"].isin(treat_filter)
            & df_conv["segment"].isin(seg_filter)
        ]

        st.dataframe(
            filtered.style.format({
                "conversion_rate": "{:.2%}",
                "cost_per_conversion": "${:.2f}",
            }),
            use_container_width=True,
            height=400,
        )

        # Aggregated view
        st.markdown("#### Aggregated by Campaign × Treatment")
        agg = (
            filtered.groupby(["campaign_name", "treatment"])
            .agg(
                total_conversions=("conversions", "sum"),
                total_impressions=("impressions", "sum"),
                avg_conv_rate=("conversion_rate", "mean"),
                avg_cpc=("cost_per_conversion", "mean"),
            )
            .reset_index()
        )
        st.dataframe(
            agg.style.format({
                "avg_conv_rate": "{:.2%}",
                "avg_cpc": "${:.2f}",
            }),
            use_container_width=True,
        )

    with sub2:
        st.markdown("#### Channel Engagement Data")

        ec1, ec2 = st.columns(2)
        chan_filter = ec1.multiselect(
            "Channel", df_eng["channel"].unique(), default=list(df_eng["channel"].unique())
        )
        ecamp_filter = ec2.multiselect(
            "Campaign", df_eng["campaign_name"].unique(), default=list(df_eng["campaign_name"].unique())
        )

        filtered_e = df_eng[
            df_eng["channel"].isin(chan_filter) & df_eng["campaign_name"].isin(ecamp_filter)
        ]

        st.dataframe(
            filtered_e.style.format({
                "spend": "${:.2f}",
                "ctr": "{:.2%}",
                "engagement_score": "{:.3f}",
            }),
            use_container_width=True,
            height=400,
        )

        # Aggregated by channel
        st.markdown("#### Aggregated by Channel")
        agg_e = (
            filtered_e.groupby("channel")
            .agg(
                total_spend=("spend", "sum"),
                total_impressions=("impressions", "sum"),
                total_clicks=("clicks", "sum"),
                avg_ctr=("ctr", "mean"),
                avg_session=("avg_session_duration", "mean"),
                avg_eng_score=("engagement_score", "mean"),
            )
            .reset_index()
        )
        st.dataframe(
            agg_e.style.format({
                "total_spend": "${:,.2f}",
                "avg_ctr": "{:.2%}",
                "avg_session": "{:.0f}s",
                "avg_eng_score": "{:.3f}",
            }),
            use_container_width=True,
        )


# ═══════════════════════════════════════════
# TAB 3 — About
# ═══════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## 📊 Campaign Analytics RAG Agent

    ### Architecture
    This is a **Retrieval-Augmented Generation (RAG)** agent built for campaign analytics demonstration.

    ```
    User Query
        │
        ▼
    ┌─────────────────────┐
    │   Query Embedding   │  ← OpenAI / Gemini embeddings
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   ChromaDB Vector   │  ← In-memory cosine similarity search
    │   Store (Retrieval) │
    └──────────┬──────────┘
               │  Top-K relevant documents
               ▼
    ┌─────────────────────┐
    │   LLM Generation    │  ← GPT-4o-mini / Gemini Flash
    │   (Context + Query) │
    └──────────┬──────────┘
               │
               ▼
          📝 Answer
    ```

    ### Data Schema
    **Conversion Table** — `campaign_id`, `campaign_name`, `treatment`, `segment`,
    `conversions`, `impressions`, `clicks`, `conversion_rate`, `cost_per_conversion`

    **Engagement Table** — `channel`, `campaign_id`, `campaign_name`, `spend`,
    `impressions`, `clicks`, `ctr`, `avg_session_duration`, `engagement_score`

    ### Vector Index Contents
    | Collection | Count | Description |
    |---|---|---|
    | Conversion rows | 100 | One chunk per row |
    | Engagement rows | 100 | One chunk per row |
    | Campaign summaries | 5 | Aggregated per-campaign facts |
    | **Total** | **205** | **Chunks in vector DB** |

    ### Tech Stack
    | Component | Technology |
    |---|---|
    | Frontend | Streamlit |
    | Vector DB | ChromaDB (in-memory) |
    | Embeddings | OpenAI `text-embedding-3-small` or Gemini `text-embedding-004` |
    | LLM | OpenAI GPT-4o-mini / Gemini Flash |
    | Deployment | Streamlit Cloud |

    ### How to Deploy on Streamlit Cloud
    1. Push all files to a GitHub repository
    2. Connect the repo to [share.streamlit.io](https://share.streamlit.io)
    3. Set main file as `app.py`
    4. No secrets needed — API key entered at runtime in the app
    """)
