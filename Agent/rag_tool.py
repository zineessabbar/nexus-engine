
from logger import get_logger
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rag')))
from Rag.retrieval import hybrid_search,BM25SearchIndex,load_all_chunks
from Rag.embeddings import load_embedding_model
from Rag.vector_store import get_db_conn
from langchain_core.tools import StructuredTool
from config import RERANK_SCORE_THRESHOLD

logger = get_logger("rag_tool")


class RAGTool:
    def __init__(self,embedding_model,cross_encoder,all_chunks,chunks_by_id,bm25_index):
        self.embedding_model=embedding_model
        self.all_chunks=all_chunks
        self.chunks_by_id=chunks_by_id
        self.bm25_index=bm25_index
        self.cross_encoder = cross_encoder
        pass

    
    def recherche_normes(self,query):
        conn=get_db_conn()
        try:
            top_chunks = hybrid_search(
                query=query, 
                conn=conn, 
                embedding_model=self.embedding_model, 
                bm25_index=self.bm25_index, 
                all_chunks=self.all_chunks, 
                chunks_by_id=self.chunks_by_id, 
                cross_encoder=self.cross_encoder
            )

            for c in top_chunks:
                logger.debug(f"Chunk: Article {c['article_number']} | score: {c['rerank_score']:.4f}")
        
            filtered_chunks=[
                c for c in top_chunks
                if c.get("rerank_score",-999.0) >= RERANK_SCORE_THRESHOLD
            ]

            if not filtered_chunks:
                return "Aucune information pertinente trouvée pour cette question."

            result_text="Voici les extraits officiels trouvés:\n\n"
            for i, chunk in enumerate(filtered_chunks,1):
                result_text += f" Source: {chunk['doc_source']}| Article {chunk['article_number']}\n"
                result_text += f"{chunk['text']}\n"
            return result_text
        finally:
            if conn:
                conn.close()
        
    def as_langgraph_tool(self):
        return StructuredTool.from_function(
                func=self.recherche_normes,
                name="recherche_normes",
                description="Permet de rechercher des normes et chartes dans le corpus.",
            )