import psycopg2
from psycopg2 import sql
from embeddings import load_embedding_model
# pyrefly: ignore [missing-import]
from pgvector.psycopg2 import register_vector  
from config import DB_CONFIG,DOCUMENTS_DIR,EMBEDDING_DIMENSION
from logger import get_logger

logger=get_logger("vector_store")
def get_db_conn():
    conn= psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def init_table(conn):
    """S'assure que l'extension et la table cible existent."""
    cur = conn.cursor()
    # Activation de l'extension vector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Création de la table chunks
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc_source VARCHAR(255),
            article_number VARCHAR(50),
            title TEXT,
            text TEXT,
            embedding vector({EMBEDDING_DIMENSION})
        );
    """)
    conn.commit()
    cur.close()

def clear_existing_chunks(conn,doc_source):
    cur=conn.cursor()
    cur.execute(
        "DELETE FROM chunks WHERE doc_source = %s",
        (doc_source,)
    )
    conn.commit()
    cur.close()

def insert_chunks(conn,chunks,doc_source,embeddings):
    cur=conn.cursor()
    for chunk,embedding in zip(chunks,embeddings):
        cur.execute("INSERT INTO chunks (doc_source,article_number,title,text, embedding)"
        "VALUES(%s,%s,%s,%s,%s)"
        ,(doc_source,chunk["article_number"],chunk["title"],chunk["text"],
        embedding.tolist(),
        )
        )
    conn.commit()
    cur.close()    

def main():
    from chunking import load_document, chunk_by_article
    import os
    
    print("Chargement du modèle d'embedding...")
    model = load_embedding_model()
    connection = get_db_conn()

    try:
        init_table(connection)
        
        fichiers = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".txt")]
        print(f"{len(fichiers)} fichiers trouvés : {fichiers}\n")

        for fichier in fichiers:
            path = os.path.join(DOCUMENTS_DIR, fichier)
            print(f"Indexation de '{fichier}'...")
            
            text = load_document(path)
            chunks = chunk_by_article(text)
            
            texts = [c["text"] for c in chunks]
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            clear_existing_chunks(connection, fichier)
            insert_chunks(connection, chunks, fichier, embeddings)
            print(f"  {len(chunks)} chunks indexés\n")

        print("Indexation batch complète.")
        
        # Vérification finale
        cur = connection.cursor()
        cur.execute("SELECT doc_source, COUNT(*) FROM chunks GROUP BY doc_source ORDER BY doc_source")
        rows = cur.fetchall()
        cur.close()
        print("\nRécapitulatif en base :")
        for row in rows:
            print(f"  {row[0]} : {row[1]} chunks")
            
    finally:
        connection.close()

if __name__ == "__main__":
    main()