from config import EMBEDDING_MODEL
from logger import get_logger
import torch
from sentence_transformers import SentenceTransformer

logger = get_logger("embeddings")

def load_embedding_model():
    logger.info(f"Chargement du modèle d'embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL,device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Modèle d'embeddings chargé avec succès")
    return model

def generate_embeddings(chunks,model):
    logger.info(f"Génération des embeddings pour {len(chunks)} chunks")
    embeddings=model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    logger.info(f"Embeddings générés avec succès")
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    return chunks

def main():
    model=load_embedding_model()
    from chunking import chunk_by_article,load_document
    text=load_document("./documents/normes.txt")
    chunks=chunk_by_article(text)
    chunks=generate_embeddings(chunks,model)

    first_chunk = chunks[0]
    print(f"Article : {first_chunk['article_number']}")
    print(f"Dimension du vecteur : {len(first_chunk['embedding'])}")
    print(f"Début du vecteur : {first_chunk['embedding'][:5]}")

if __name__ == "__main__":
    main()