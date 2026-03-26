from __future__ import annotations

import os
from openai import OpenAI

from config import HF_MODEL_ID, HF_TOKEN


def get_hf_client() -> OpenAI:
    """Create OpenAI-compatible client for HF Router."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required for HF Router API")
    
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


def call_llama_via_router(query: str, context: str, max_tokens: int = 512) -> str:
    """Call Llama 3.3-70B (or other) via HF Router + OpenAI client."""
    client = get_hf_client()

    system_prompt = (
        "You are an analytics assistant for marketing campaigns. "
        "You are given structured context with conversion and channel engagement metrics. "
        "Use ONLY the provided context to answer, be concise and data-driven, and clearly state "
        "which campaigns, treatments, or channels you are referring to."
    )

    user_prompt = (
        f"Question: {query}\n\n"
        f"Here is the campaign data context (conversion + engagement):\n\n{context}\n\n"
        "Now answer the question step by step, then give a short final conclusion."
    )

    try:
        completion = client.chat.completions.create(
            model=HF_MODEL_ID,  # e.g., "meta-llama/Llama-3.3-70B-Instruct:groq"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        message = completion.choices[0].message
        return message.content or "(No content returned)"

    except Exception as e:
        return f"LLM call failed: {e}"
