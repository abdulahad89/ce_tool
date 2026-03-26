import os

# ✅ Llama 3.3 70B via HF Router (Groq-powered, super fast!)
HF_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct:groq"  
HF_TOKEN = os.environ.get("HF_TOKEN")  # REQUIRED for router

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CONVERSIONS_PATH = os.path.join(BASE_DIR, "data", "sample_conversions.csv")
DATA_ENGAGEMENT_PATH = os.path.join(BASE_DIR, "data", "sample_engagement.csv")

TOP_K_DEFAULT = 4
