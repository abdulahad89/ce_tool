import textwrap, os

base_dir = 'output/gemini_openai_llm_client'
os.makedirs(base_dir, exist_ok=True)

# Universal llm_client.py - supports HF, Gemini, OpenAI
universal_llm = '''
from __future__ import annotations

import os
from typing import Optional

# HF Router (OpenAI-compatible)
from openai import OpenAI
from huggingface_hub import InferenceClient

# Google Gemini
import google.generativeai as genai

from config import LLM_PROVIDER, HF_MODEL_ID, HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY


def get_client(provider: str) -> OpenAI | genai.GenerativeModel | InferenceClient:
    """Get client for specified provider."""
    if provider == "hf_router":
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN required for HF Router")
        return OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required")
        return OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY required")
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-1.5-pro')  # or 'gemini-1.5-flash'
    elif provider == "hf_direct":
        if HF_TOKEN:
            return InferenceClient(api_key=HF_TOKEN)
        return InferenceClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(query: str, context: str, max_tokens: int = 512, provider: str = "hf_router") -> str:
    """Universal LLM call - supports HF Router, OpenAI, Gemini, HF Direct."""
    client = get_client(provider)

    system_prompt = (
        "You are an analytics assistant for marketing campaigns. "
        "You are given structured context with conversion and channel engagement metrics. "
        "Use ONLY the provided context to answer, be concise and data-driven, and clearly state "
        "which campaigns, treatments, or channels you are referring to."
    )

    user_prompt = (
        f"Question: {query}\\n\\n"
        f"Here is the campaign data context (conversion + engagement):\\n\\n{context}\\n\\n"
        "Now answer the question step by step, then give a short final conclusion."
    )

    try:
        if isinstance(client, (OpenAI, genai.GenerativeModel)):  # OpenAI/Gemini
            if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):  # OpenAI
                completion = client.chat.completions.create(
                    model=HF_MODEL_ID if provider == "hf_router" else None,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return completion.choices[0].message.content or "(No content)"
            else:  # Gemini
                full_prompt = system_prompt + "\\n\\n" + user_prompt
                response = client.generate_content(full_prompt, generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                })
                return response.text or "(No content)"
        else:  # HF InferenceClient
            completion = client.chat.completions.create(
                model=HF_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            message = completion.choices[0].message
            if isinstance(message, dict):
                return message.get("content", "(No content)")
            return str(message)

    except Exception as e:
        return f"{provider.upper()} call failed: {e}"
'''

with open(os.path.join(base_dir, 'llm_client.py'), 'w') as f:
    f.write(textwrap.dedent(universal_llm))
