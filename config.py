import os

# Hugging Face model + token
HF_MODEL_ID: str = os.environ.get("HF_MODEL_ID", "mistralai/Mistral-Nemo-Instruct-2407")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")

# Embedding model for retrieval
EMBEDDING_MODEL_NAME: str = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CONVERSIONS_PATH = os.path.join(BASE_DIR, "data", "sample_conversions.csv")
DATA_ENGAGEMENT_PATH = os.path.join(BASE_DIR, "data", "sample_engagement.csv")

TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", 4))
