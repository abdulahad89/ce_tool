from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATA_CONVERSIONS_PATH,
    DATA_ENGAGEMENT_PATH,
    EMBEDDING_MODEL_NAME,
)

_embedding_model: SentenceTransformer | None = None
_conv_df: pd.DataFrame | None = None
_eng_df: pd.DataFrame | None = None
_docs: list[str] | None = None
_doc_meta: list[dict] | None = None
_emb_matrix: np.ndarray | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _conv_df, _eng_df
    if _conv_df is None or _eng_df is None:
        _conv_df = pd.read_csv(DATA_CONVERSIONS_PATH)
        _eng_df = pd.read_csv(DATA_ENGAGEMENT_PATH)
    return _conv_df, _eng_df


def build_docs_and_embeddings() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[dict]]:
    """Load data, build human-readable docs, and compute embeddings."""
    global _docs, _doc_meta, _emb_matrix

    conv_df, eng_df = load_tables()

    if _docs is not None and _emb_matrix is not None and _doc_meta is not None:
        return conv_df, eng_df, _emb_matrix, _docs, _doc_meta

    # Build conversion documents
    conv_docs: list[str] = []
    for _, row in conv_df.iterrows():
        doc = (
            f"[CONVERSION] Campaign {row['campaign_name']} ({row['campaign_id']}) "
            f"- Treatment {row['treatment']} for segment {row['segment']}. "
            f"Conversions: {row['conversions']} from {row['impressions']} impressions and "
            f"{row['clicks']} clicks (CR={row['conversion_rate']:.2%}, "
            f"CPC={row['cost_per_conversion']:.2f})."
        )
        conv_docs.append(doc)

    # Build engagement documents
    eng_docs: list[str] = []
    for _, row in eng_df.iterrows():
        doc = (
            f"[ENGAGEMENT] {row['channel']} for campaign {row['campaign_name']} "
            f"({row['campaign_id']}). Spend {row['spend']:.2f}, impressions {row['impressions']}, "
            f"clicks {row['clicks']} (CTR={row['ctr']:.2%}). "
            f"Avg session {row['avg_session_duration']}s, engagement score {row['engagement_score']:.2f}."
        )
        eng_docs.append(doc)

    docs: list[str] = conv_docs + eng_docs
    doc_meta: list[dict] = [
        {"type": "conversion", "row_index": i}
        for i in range(len(conv_docs))
    ] + [
        {"type": "engagement", "row_index": i}
        for i in range(len(eng_docs))
    ]

    model = get_embedding_model()
    emb_matrix = model.encode(docs, normalize_embeddings=True)

    _docs = docs
    _doc_meta = doc_meta
    _emb_matrix = np.array(emb_matrix, dtype=np.float32)

    return conv_df, eng_df, _emb_matrix, _docs, _doc_meta


def retrieve(query: str, top_k: int = 4) -> tuple[str, list[int]]:
    """Retrieve top-K docs as text context + indices using cosine similarity."""
    conv_df, eng_df, emb_matrix, docs, meta = build_docs_and_embeddings()

    model = get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)

    sims = cosine_similarity(q_emb, emb_matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    context_chunks: list[str] = []
    for rank, idx in enumerate(top_idx, start=1):
        m = meta[idx]
        sim = sims[idx]
        header = f"[Rank {rank} | Type={m['type']} | Score={sim:.3f}]"
        context_chunks.append(header + "\n" + docs[idx])

    context = "\n\n".join(context_chunks)
    return context, top_idx.tolist()
