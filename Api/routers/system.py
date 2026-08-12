from fastapi import APIRouter,HTTPException
from Rag.vector_store import get_db_conn

from Agent.agent_tool import init_rag_tool
from Agent.graph import build_workflow
import Api.routers.audit as audit_module

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

@router.post("/reload")
async def reload_corpus():
    try:
        import asyncio
        def _reload():
            new_rag_tool=init_rag_tool()
            new_workflow=build_workflow(new_rag_tool)
            audit_module.new_workflow= new_workflow
            return True

        await asyncio.to_thread(_reload)
        return {"status":"reload_success","message":"Corpus rechargé avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500,detail={"status":"reload_error","error":str(e)})