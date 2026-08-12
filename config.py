# config.py
from pathlib import Path
from dotenv import load_dotenv
import os 
load_dotenv()

# --- Chemins ---
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "Rag" / "documents"

# --- Base de données ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST","localhost"),
    "port": int(os.getenv("DB_PORT",5432)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME","rag_db"),
}

# --- Modèles ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen2.5:7b-instruct"
LLM_TEMPERATURE = 0

# --- Retrieval ---
CANDIDATES_PER_METHOD = 10
CANDIDATES_TO_RERANK = 10
RERANK_SCORE_THRESHOLD = 0.01
FINAL_TOP_K = 4

# --- API ---
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]

# --- LLM Providers ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AVAILABLE_MODELS = {
    "qwen2.5:7b": {
        "provider": "ollama",
        "label": "Qwen 2.5 7B (Local)",
        "description": "Modèle local — souveraineté des données garantie",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "label": "GPT-4o Mini (OpenAI)",
        "description": "Modèle cloud — données envoyées vers OpenAI",
    },
    "gpt-4o": {
        "provider": "openai",
        "label": "GPT-4o (OpenAI)",
        "description": "Modèle cloud haute performance",
    },
}