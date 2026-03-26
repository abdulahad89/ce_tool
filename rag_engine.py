"""
RAG Engine for Campaign Analytics Agent.
Uses:
- Local embeddings (Sentence Transformers) → FREE + stable
- OpenAI OR Gemini for generation
- ChromaDB as vector store
"""

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

        name = crows[0]["campaign_name"] if crows else erows[0]["campaign_name"]

        total_conv = sum(int(r["conversions"]) for r in crows)
        total_imp = sum(int(r["impressions"]) for r in crows)
        total_clicks = sum(int(r["clicks"]) for r in crows)

        total_spend = sum(float(r["spend"]) for r in erows)
        avg_ctr = (sum(float(r["ctr"]) for r in erows) / len(erows)) if erows else 0

        text = (
            f"SUMMARY — Campaign '{name}' (ID: {cid}): "
            f"Conversions: {total_conv}, Impressions: {total_imp}, Clicks: {total_clicks}. "
            f"Spend: ${total_spend:.2f}, Avg CTR: {avg_ctr:.2%}."
        )

        meta = {
            "source": "summary",
            "campaign_id": cid,
            "campaign_name": name,
        }

        summaries.append((text, meta))

    return summaries


# ─────────────────────────────────────────────────────────────
# 🆓 LOCAL EMBEDDINGS (MAIN FIX)
# ─────────────────────────────────────────────────────────────

def _embed_local(texts: List[str]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer

    if not hasattr(_embed_local, "model"):
        _embed_local.model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embed_local.model.encode(texts).tolist()


# ─────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self, api_key: str, provider: str, model: str, top_k: int = 5):
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.top_k = top_k

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="campaign_data",
            metadata={"hnsw:space": "cosine"},
        )

        self._build_index()

    def _build_index(self):
        conv_rows = _parse_csv(CONVERSIONS_CSV)
        eng_rows = _parse_csv(ENGAGEMENT_CSV)

        docs = (
            _build_conversion_docs(conv_rows)
            + _build_engagement_docs(eng_rows)
            + _build_campaign_summaries(conv_rows, eng_rows)
        )

        texts = [d[0] for d in docs]
        metadatas = [d[1] for d in docs]
        ids = [str(uuid.uuid4()) for _ in docs]

        embeddings = _embed_local(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query: str) -> List[str]:
        q_emb = _embed_local([query])[0]

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
        )

        return results["documents"][0]

    def generate(self, query: str, context_docs: List[str]) -> str:
        context = "\n\n".join(context_docs)

        system_prompt = "You are a campaign analytics expert."

        user_prompt = f"""
Context:
{context}

Question: {query}
"""

        if self.provider == "OpenAI":
            return self._generate_openai(system_prompt, user_prompt)
        else:
            return self._generate_gemini(system_prompt, user_prompt)

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )

        return response.text

    def query(self, question: str):
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        return answer, docs
