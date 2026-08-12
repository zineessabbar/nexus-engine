
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rag')))
import torch
from sentence_transformers import CrossEncoder

from Rag.vector_store import get_db_conn
from Rag.retrieval import load_all_chunks, BM25SearchIndex
from Rag.embeddings import load_embedding_model
from Agent.rag_tool import RAGTool

from config import RERANKER_MODEL
from logger import get_logger

logger=get_logger("agent_tool")
_rag_tool = None


def init_rag_tool() -> RAGTool:
    global _rag_tool
    logger.info("Initialisation du RAG tool")

    embedding_model = load_embedding_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = CrossEncoder(RERANKER_MODEL, device=device)

    init_conn = get_db_conn()
    try:
        all_chunks = load_all_chunks(init_conn)
    finally:
        init_conn.close()

    chunks_by_id = {c["id"]: c for c in all_chunks}
    bm25_index = BM25SearchIndex(all_chunks)

    _rag_tool = RAGTool(
        embedding_model=embedding_model,
        cross_encoder=cross_encoder,
        bm25_index=bm25_index,
        all_chunks=all_chunks,
        chunks_by_id=chunks_by_id,
    )

    logger.info("RAG tool initialisé")
    return _rag_tool


def get_rag_tool() -> RAGTool:
    return _rag_tool