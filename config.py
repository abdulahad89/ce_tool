import os

# Provider selection - change this to switch LLMs
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini", "hf_router", "openai", "hf_direct"

# Google Gemini (FREE - get at https://makersuite.google.com/app/apikey)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_gemini_key_here")

# OpenAI (PAID)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# HF Router (FREE credits)
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct:groq")
HF_TOKEN = os.environ.get("HF_TOKEN")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CONVERSIONS_PATH = os.path.join(BASE_DIR, "data", "sample_conversions.csv")
DATA_ENGAGEMENT_PATH = os.path.join(BASE_DIR, "data", "sample_engagement.csv")

TOP_K_DEFAULT = 4
