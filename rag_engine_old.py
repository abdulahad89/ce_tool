"""
RAG Engine for Campaign Analytics Agent.
Supports OpenAI and Google Gemini for embeddings + generation.
Uses ChromaDB as the in-memory vector store.
"""

import re
import csv
import io
import uuid
import chromadb
from typing import List, Tuple
from sample_data import CONVERSIONS_CSV, ENGAGEMENT_CSV


# ─────────────────────────────────────────────────────────────
# Document builders
# ─────────────────────────────────────────────────────────────

def _parse_csv(csv_str: str) -> List[dict]:
    reader = csv.DictReader(io.StringIO(csv_str.strip()))
    return list(reader)


def _build_conversion_docs(rows: List[dict]) -> List[Tuple[str, dict]]:
    """Create rich text chunks from conversion rows."""
    docs = []
    for r in rows:
        text = (
            f"Campaign '{r['campaign_name']}' (ID: {r['campaign_id']}) | "
            f"Treatment: {r['treatment']} | Segment: {r['segment']} | "
            f"Conversions: {r['conversions']} | Impressions: {r['impressions']} | "
            f"Clicks: {r['clicks']} | "
            f"Conversion Rate: {float(r['conversion_rate']):.2%} | "
            f"Cost Per Conversion: ${float(r['cost_per_conversion']):.2f}"
        )
        meta = {
            "source": "conversions",
            "campaign_id": r["campaign_id"],
            "campaign_name": r["campaign_name"],
            "treatment": r["treatment"],
            "segment": r["segment"],
        }
        docs.append((text, meta))
    return docs


def _build_engagement_docs(rows: List[dict]) -> List[Tuple[str, dict]]:
    """Create rich text chunks from engagement rows."""
    docs = []
    for r in rows:
        text = (
            f"Channel '{r['channel']}' for campaign '{r['campaign_name']}' (ID: {r['campaign_id']}) | "
            f"Spend: ${float(r['spend']):.2f} | Impressions: {r['impressions']} | "
            f"Clicks: {r['clicks']} | CTR: {float(r['ctr']):.2%} | "
            f"Avg Session Duration: {r['avg_session_duration']}s | "
            f"Engagement Score: {float(r['engagement_score']):.3f}"
        )
        meta = {
            "source": "engagement",
            "channel": r["channel"],
            "campaign_id": r["campaign_id"],
            "campaign_name": r["campaign_name"],
        }
        docs.append((text, meta))
    return docs


def _build_campaign_summaries(conv_rows: List[dict], eng_rows: List[dict]) -> List[Tuple[str, dict]]:
    """Aggregate per-campaign summaries for higher-level queries."""
    from collections import defaultdict
    camp_conv = defaultdict(list)
    for r in conv_rows:
        camp_conv[r["campaign_id"]].append(r)

    camp_eng = defaultdict(list)
    for r in eng_rows:
        camp_eng[r["campaign_id"]].append(r)

    summaries = []
    all_ids = set(list(camp_conv.keys()) + list(camp_eng.keys()))

    for cid in all_ids:
        crows = camp_conv.get(cid, [])
        erows = camp_eng.get(cid, [])
        name = crows[0]["campaign_name"] if crows else erows[0]["campaign_name"] if erows else cid

        # Conversion stats
        total_conv = sum(int(r["conversions"]) for r in crows)
        total_imp_c = sum(int(r["impressions"]) for r in crows)
        total_clicks_c = sum(int(r["clicks"]) for r in crows)
        avg_cpc = (sum(float(r["cost_per_conversion"]) for r in crows) / len(crows)) if crows else 0
        treatments = list({r["treatment"] for r in crows})
        segments = list({r["segment"] for r in crows})

        # Engagement stats
        total_spend = sum(float(r["spend"]) for r in erows)
        avg_ctr = (sum(float(r["ctr"]) for r in erows) / len(erows)) if erows else 0
        avg_eng = (sum(float(r["engagement_score"]) for r in erows) / len(erows)) if erows else 0
        channels = list({r["channel"] for r in erows})

        text = (
            f"SUMMARY — Campaign '{name}' (ID: {cid}): "
            f"Total Conversions: {total_conv} across {total_imp_c:,} impressions "
            f"({total_clicks_c:,} clicks). Avg Cost Per Conversion: ${avg_cpc:.2f}. "
            f"Treatments tested: {', '.join(treatments)}. Segments: {', '.join(segments)}. "
            f"Total Spend: ${total_spend:,.2f}. Avg CTR: {avg_ctr:.2%}. "
            f"Avg Engagement Score: {avg_eng:.3f}. "
            f"Channels used: {', '.join(channels)}."
        )
        meta = {
            "source": "summary",
            "campaign_id": cid,
            "campaign_name": name,
        }
        summaries.append((text, meta))

    return summaries


# ─────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────

def _embed_openai(texts: List[str], api_key: str) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# def _embed_gemini(texts: List[str], api_key: str) -> List[List[float]]:
#     #import google.generativeai as genai
#     from google import genai
#     genai.configure(api_key=api_key)
#     embeddings = []
#     for text in texts:
#         result = genai.embed_content(
#             model="models/embedding-001",
#             content=text,
#             task_type="retrieval_document",
#         )
#         embeddings.append(result["embedding"])
#     return embeddings

from typing import List
def _embed_gemini(texts: List[str], api_key: str) -> List[List[float]]:
    from google import genai

    client = genai.Client(api_key=api_key)

    response = client.models.embed_content(
        model="embedding-001",   # ✅ correct model name
        contents=texts           # ✅ batch input
    )

    return [e.values for e in response.embeddings]


# ─────────────────────────────────────────────────────────────
# Main RAG Engine
# ─────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self, api_key: str, provider: str, model: str, top_k: int = 5):
        self.api_key = api_key
        self.provider = provider  # "OpenAI" or "Gemini"
        self.model = model
        self.top_k = top_k

        # In-memory ChromaDB
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="campaign_data",
            metadata={"hnsw:space": "cosine"},
        )
        self._build_index()

    def _build_index(self):
        """Parse data, embed, and store in ChromaDB."""
        conv_rows = _parse_csv(CONVERSIONS_CSV)
        eng_rows = _parse_csv(ENGAGEMENT_CSV)

        conv_docs = _build_conversion_docs(conv_rows)
        eng_docs = _build_engagement_docs(eng_rows)
        summary_docs = _build_campaign_summaries(conv_rows, eng_rows)

        all_docs = conv_docs + eng_docs + summary_docs
        texts = [d[0] for d in all_docs]
        metadatas = [d[1] for d in all_docs]
        ids = [str(uuid.uuid4()) for _ in all_docs]

        # Batch embed
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self.provider == "OpenAI":
                batch_emb = _embed_openai(batch, self.api_key)
            else:
                batch_emb = _embed_gemini(batch, self.api_key)
            all_embeddings.extend(batch_emb)

        self.collection.add(
            documents=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query: str) -> List[str]:
        """Embed query and retrieve top-k documents."""
        if self.provider == "OpenAI":
            q_emb = _embed_openai([query], self.api_key)[0]
        else:
            q_emb = _embed_gemini([query], self.api_key)[0]

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
        )
        return results["documents"][0]  # list of strings

    def generate(self, query: str, context_docs: List[str]) -> str:
        """Generate an answer using the LLM with retrieved context."""
        context = "\n\n".join(f"[Doc {i+1}]: {doc}" for i, doc in enumerate(context_docs))

        system_prompt = """You are an expert Campaign Analytics AI agent. 
You have access to two datasets:
1. **Conversion Data** — campaign-level conversion metrics (conversions, impressions, clicks, conversion rate, cost per conversion) broken down by treatment variant and customer segment.
2. **Channel Engagement Data** — channel-level engagement metrics (spend, impressions, clicks, CTR, avg session duration, engagement score) for each campaign.

Use ONLY the provided context documents to answer questions. 
When relevant, compare treatments (Control vs Variant A/B/Optimized), analyze segments, and highlight channel performance.
Be specific with numbers and metrics. If the context doesn't have enough data, say so clearly.
Format your answers clearly with bullet points or tables where helpful."""

        user_message = f"""Context Documents:
{context}

Question: {query}

Please analyze the context and provide a detailed, data-driven answer."""

        if self.provider == "OpenAI":
            return self._generate_openai(system_prompt, user_message)
        else:
            return self._generate_gemini(system_prompt, user_message)

    def _generate_openai(self, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def _generate_gemini(self, system_prompt: str, user_message: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        gemini_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )
        response = gemini_model.generate_content(
            user_message,
            generation_config={"temperature": 0.2, "max_output_tokens": 1500},
        )
        return response.text

    def query(self, user_question: str) -> Tuple[str, List[str]]:
        """Full RAG pipeline: retrieve + generate."""
        docs = self.retrieve(user_question)
        answer = self.generate(user_question, docs)
        return answer, docs
