import torch
from sentence_transformers import CrossEncoder
from config import (CANDIDATES_PER_METHOD,CANDIDATES_TO_RERANK,RERANK_SCORE_THRESHOLD,FINAL_TOP_K,RERANKER_MODEL)
import re
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from logger import get_logger

logger = get_logger("retreival")

def load_all_chunks(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, article_number, title, text, doc_source FROM chunks")
    rows = cur.fetchall()
    cur.close()
    return [
        {
            "id": row[0], 
            "article_number": row[1], 
            "title": row[2], 
            "text": row[3],
            "doc_source": row[4] 
        } for row in rows
    ]

def vector_search_ranking(conn,query_embedding,top_k):
    embedding_str=query_embedding.tolist()
    cur=conn.cursor()
    cur.execute("""SELECT id FROM chunks ORDER BY embedding <=>%s :: vector ASC LIMIT %s""",(embedding_str,top_k))
    rows=cur.fetchall()
    cur.close()
    return [row[0] for row in rows]

def hybrid_search(query,conn,embedding_model,cross_encoder,bm25_index,all_chunks,chunks_by_id):
    logger.debug(f"Recherche hybride pour:'{query[:60]}'")
    query_embedding=embedding_model.encode([query],normalize_embeddings=True)[0]
    vector_topk=vector_search_ranking(conn,query_embedding,top_k=CANDIDATES_PER_METHOD)

    bm25_results=bm25_index.search(query,top_k=CANDIDATES_PER_METHOD)
    bm25_ids=[all_chunks[idx]['id'] for idx,score in bm25_results]

    fused = fuse_rankings(vector_ranking = vector_topk,bm25_ranking=bm25_ids)
    candidates = [chunks_by_id[chunk_id] for chunk_id,_score in fused[:CANDIDATES_TO_RERANK]]
    if not candidates:
        return []
    
    pairs = [(query,c["text"]) for c in candidates]
    scores=cross_encoder.predict(pairs)
    scored = list(zip(candidates,scores))
    scored.sort(key=lambda x:x[1],reverse=True)

    top_chunks = [{**chunk, "rerank_score":float(score)} for chunk,score in scored[:FINAL_TOP_K]]
    logger.info(f"Retrieval terminé-{len(top_chunks)} chunks récupérés")
    return top_chunks

def fuse_rankings(vector_ranking,bm25_ranking,k=60):
    rrf_scores={}
    for rank,chunk_id in enumerate(vector_ranking,1):
        rrf_scores[chunk_id]=rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank,chunk_id in enumerate(bm25_ranking,1):
        rrf_scores[chunk_id]=rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(),key=lambda x:x[1],reverse=True)
    
def rerank(query,candidates,cross_encoder,top_k):
    pairs=[(query,c["text"])for c in candidates]
    scores=cross_encoder.predict(pairs)
    scored_candidates=list(zip(candidates,scores))
    scored_candidates.sort(key=lambda x:x[1],reverse=True)
    return [c for c,_score in scored_candidates[:top_k]]

def tokenize(text):
    text=text.lower()
    return re.findall(r"\b\w{2,}\b",text)

class BM25SearchIndex:
    def __init__(self,chunks):
        self.chunks=chunks
        tokenized_docs=[tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self,query,top_k):
        tokenized_query=tokenize(query)
        scores=self.bm25.get_scores(tokenized_query)
        ranked=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
        return ranked[:top_k]

def main():
    import numpy as np
    from embeddings import load_embedding_model
    from vector_store import get_db_conn
    from sentence_transformers import CrossEncoder
    import torch

    print("Connexion à PostgreSQL...")
    connection = get_db_conn()

    print("1. Pipeline Aval : Chargement des chunks depuis pgvector...")
    chunks = load_all_chunks(connection)
    chunks_by_id = {c['id']: c for c in chunks}

    print("2. Pipeline Aval : Création de l'index BM25...")
    bm25_index = BM25SearchIndex(chunks)

    print("Chargement du modèle d'embeddings...")
    embedding_model = load_embedding_model()
    
    print("Chargement du modèle de Reranking (Cross-Encoder)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    query = "Quelle est la longueur minimale pour les mots de passe ?"
    print(f"\n3. Requête de recherche : {query}")

    candidates = hybrid_search(query, connection, embedding_model, cross_encoder, bm25_index, chunks, chunks_by_id)


    print("\n Top 3 résultats finaux :")
    for i, chunk in enumerate(candidates[:3], 1):
        print(f"\n{i}. Article : {chunk['article_number']}")
        print(f"   Titre : {chunk['title']}")
        print(f"   Texte : {chunk['text']}")
        
    connection.close()

if __name__ == "__main__":
    main()