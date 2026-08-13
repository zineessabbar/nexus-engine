import psycopg2
from psycopg2 import sql
import os
from pathlib import Path
from Rag.embeddings import load_embedding_model
# pyrefly: ignore [missing-import]
from pgvector.psycopg2 import register_vector  
from config import DB_CONFIG, DOCUMENTS_DIR, EMBEDDING_DIMENSION
from logger import get_logger
from Rag.document_loader import load_and_chunk
import json

logger = get_logger("vector_store")

SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".doc"]

def get_db_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def init_table(conn):
    """S'assure que l'extension et la table cible existent avec le nouveau schéma (incluant chunk_type)."""
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc_source VARCHAR(255),
            article_number VARCHAR(50),
            title TEXT,
            text TEXT,
            chunk_type VARCHAR(50),
            embedding vector({EMBEDDING_DIMENSION})
        );
    """)
    conn.commit()
    cur.close()

def clear_existing_chunks(conn, doc_source):
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM chunks WHERE doc_source = %s",
        (doc_source,)
    )
    cur.close()

def insert_chunks(conn, chunks, doc_source, embeddings):
    cur = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        # Insertion incluant le champ 'chunk_type' issu de ton nouveau document_loader
        cur.execute(
            """INSERT INTO chunks (doc_source, article_number, title, text, chunk_type, embedding)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                doc_source, 
                chunk["article_number"], 
                chunk["title"], 
                chunk["text"], 
                chunk.get("chunk_type", "unknown"),
                embedding.tolist()
            )
        )
    cur.close()

def main():
    print("Chargement du modèle d'embedding...")
    model = load_embedding_model()
    connection = get_db_conn()

    try:
        cur = connection.cursor()
        print("Nettoyage de l'ancienne table pour mise à jour du schéma...")
        cur.execute("DROP TABLE IF EXISTS chunks;")
        connection.commit()
        cur.close()

        init_table(connection)
        
        fichiers = [
            f for f in os.listdir(str(DOCUMENTS_DIR))
            if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
        ]
        logger.info(f"{len(fichiers)} fichiers trouvés")

        CACHE_DIR = DOCUMENTS_DIR / ".cache"
        CACHE_DIR.mkdir(exist_ok=True)

        for fichier in fichiers:
            path = DOCUMENTS_DIR / fichier
            cache_path = CACHE_DIR / f"{fichier}_cache.json" # Fichier de sauvegarde
            logger.info(f"Indexation de '{fichier}'...")
            
            # --- SYSTÈME DE CACHE ---
            if cache_path.exists():
                logger.info(f"📦 Cache trouvé ! Chargement instantané des chunks depuis {cache_path.name}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
            else:
                logger.info("⏳ Aucun cache trouvé. Lancement de l'extraction et du chunking...")
                chunks = load_and_chunk(path, embedding_model=model)
                
                # Sauvegarde immédiate dans le cache pour les prochaines fois
                if chunks:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    logger.info("💾 Chunks sauvegardés dans le cache.")
            # ------------------------
            
            if not chunks:
                logger.warning(f"Aucun chunk généré pour '{fichier}' — ignoré")
                continue
            
            # Coupe drastique à 4000 caractères pour empêcher le modèle de s'étouffer
            texts = [c["text"][:4000] for c in chunks]
            print(f"⏳ Début de la vectorisation de {len(texts)} chunks...")
            
            embeddings = model.encode(
                texts, 
                batch_size=4, 
                show_progress_bar=True, 
                normalize_embeddings=True
            )
            
            try:
                clear_existing_chunks(connection, fichier)
                insert_chunks(connection, chunks, fichier, embeddings)
                connection.commit()
                logger.info(f"'{fichier}' → {len(chunks)} chunks indexés")
            except Exception as e:
                connection.rollback()
                logger.error(f"Erreur lors de l'indexation de '{fichier}' : {e}")
                continue

        logger.info("Indexation batch complète")
        
        cur = connection.cursor()
        cur.execute(
            "SELECT doc_source, COUNT(*) FROM chunks GROUP BY doc_source ORDER BY doc_source"
        )
        rows = cur.fetchall()
        cur.close()
        print("\nRécapitulatif en base :")
        for row in rows:
            print(f"  {row[0]} : {row[1]} chunks")
            
    finally:
        connection.close()

if __name__ == "__main__":
    main()