
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

def init_rag_tool():
    logger.info("Initialisation du RAG tool")
    init_conn=get_db_conn()
    try:
        all_chunks=load_all_chunks(init_conn)
    finally:
        init_conn.close()
        


    chunks_by_id={c['id']: c for c in all_chunks}
    bm25_index=BM25SearchIndex(all_chunks)
    embedding_model=load_embedding_model()
    device="cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = CrossEncoder(RERANKER_MODEL, device=device)

    rag_tool=RAGTool(embedding_model,cross_encoder,all_chunks,chunks_by_id,bm25_index)
    logger.info("RAG tool initialisé")

    return rag_tool