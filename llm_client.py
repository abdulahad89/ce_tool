from __future__ import annotations

import os
from typing import Optional

# HF Router + OpenAI
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
