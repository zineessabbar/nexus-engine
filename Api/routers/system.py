from fastapi import APIRouter
from Rag.vector_store import get_db_conn

router=APIRouter()

@router.get("/health")
async def health_check():
    try:
        conn=get_db_conn()
        conn.close()
        db_status="connecté"
    except:
        db_status="déconnecté"
    return {
        "status":"online",
        "engine":"RAG",
        "version":"1.0",
        "db_status":db_status
    }

@router.get("/stats")
async def get_system_stats():
    try:
        conn=get_db_conn()
        try:
            cursor=conn.cursor()
            cursor.execute("SELECT count(*) FROM chunks")
            total_chunks=cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT doc_source) FROM chunks")
            total_doc=cursor.fetchone()[0]
            cursor.execute("SELECT MAX(date_audit) FROM audit_history")
            last_audit=cursor.fetchone()[0]
            cursor.close()
        finally:
            conn.close()
        return {
            "documents_indexes":total_doc,
            "chunks_indexes":total_chunks,
            "derniere_audit":last_audit.strftime("%Y-%m-%d %H:%M") if last_audit else "Aucun",
            "etat_base_de_donnees":"connecté"
        }
    except Exception as e:
        return {
            "documents_indexes":0,
            "chunks_indexes":0,
            "derniere_audit":"Inconnu",
            "etat_base_de_donnees":f"erreur:{str(e)}"
        }