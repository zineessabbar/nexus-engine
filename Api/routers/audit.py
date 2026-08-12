from Agent.agent_tool import get_rag_tool
import uuid
from datetime import datetime
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
from fastapi import StreamingResponse
from llm_providers import get_provider
from Agent.graph import build_workflow_with_provider
from config import AVAILABLE_MODELS, OPENAI_API_KEY
from Agent.agent_tool import init_rag_tool
from Agent.graph import build_workflow
from Rag.vector_store import get_db_conn
from logger import get_logger

workflow=None
logger = get_logger("api.audit")

def init_history_table():
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_history (
                id SERIAL PRIMARY KEY,
                titre_projet TEXT,
                date_audit TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                extrait TEXT,
                rapport_complet TEXT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Erreur création table d'historique : {e}")


@asynccontextmanager
async def router_lifespan(router:APIRouter):
    global workflow
    init_history_table()
    try:
        outil_rag=init_rag_tool()
        workflow =build_workflow(outil_rag)
        logger.info("Workflow LangGraph initialisé")
    except Exception as e:
        logger.error(f"Erreur los de l'initialisation du moteur:{e}")
        workflow=None
    yield

router = APIRouter(lifespan=router_lifespan)

class AuditResponse(BaseModel):
    domaines_identifies:list[str]
    rapport_final:str

class ProjectRequest(BaseModel):
    description: str
    model: str = "qwen2.5:7b"

@router.post("/analyse")
async def analyse_project(request: ProjectRequest):
    if not workflow:
        raise HTTPException(status_code=503, detail="LangGraph non initialisé")

    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu. Disponibles : {list(AVAILABLE_MODELS.keys())}"
        )

    model_config = AVAILABLE_MODELS[request.model]

    if model_config["provider"] == "openai" and not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Clé API OpenAI non configurée")

    try:
        llm_provider = get_provider(
            provider_name=model_config["provider"],
            model=request.model,
        )

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "description_projet": request.description,
            "domaines_identifies": [],
            "textes_normatifs": [],
            "rapport_final": "",
        }

        def run():
            dynamic_workflow = build_workflow_with_provider(
                get_rag_tool(), llm_provider
            )
            return dynamic_workflow.invoke(state, config)

        final_state = await asyncio.to_thread(run)

        # Persistance en base
        rapport = final_state.get("rapport_final", "")
        domaines = final_state.get("domaines_identifies", [])
        extrait = rapport[:200] if rapport else ""

        conn = get_db_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO audit_history
                    (titre_projet, date_audit, status, extrait, rapport_complet)
                VALUES (%s, NOW(), %s, %s, %s)
                """,
                (request.description[:80], "Terminé", extrait, rapport),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

        return AuditResponse(
            domaines_identifies=domaines,
            rapport_final=rapport,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur analyse : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyse/stream")
async def analyse_project_stream(request: ProjectRequest):
    # --- Validation du modèle (miroir de /analyse) ---
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu. Disponibles : {list(AVAILABLE_MODELS.keys())}"
        )

    model_config = AVAILABLE_MODELS[request.model]

    if model_config["provider"] == "openai" and not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Clé API OpenAI non configurée")

    # --- Construction dynamique du provider et du workflow ---
    try:
        llm_provider = get_provider(
            provider_name=model_config["provider"],
            model=request.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dynamic_workflow = build_workflow_with_provider(get_rag_tool(), llm_provider)

    async def generate():
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            state = {
                "description_projet": request.description,
                "domaines_identifies": [],
                "textes_normatifs": [],
                "rapport_final": ""
            }

            def run_workflow():
                return dynamic_workflow.invoke(state, config)

            final_state = await asyncio.to_thread(run_workflow)

            domaines = final_state.get("domaines_identifies", [])
            rapport = final_state.get("rapport_final", "")

            yield f"data: {json.dumps({'type': 'domaines', 'data': domaines})}\n\n"

            words = rapport.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
                await asyncio.sleep(0.02)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Erreur stream : {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )



@router.get("/history")
async def get_audit_history():
    try:
        conn=get_db_conn()
        try:
            cursor=conn.cursor()
            cursor.execute("""SELECT id,titre_projet,date_audit,rapport_final FROM audit_history ORDER BY date_audit DESC """)
            lignes=cursor.fetchall()
            cursor.close()
        finally:
            conn.close()
 
        historique=[]
        for ligne in lignes:
            historique.append({
                "id":ligne[0],
                "titre_projet":ligne[1],
                "date_audit":ligne[2].strftime("%Y-%m-%d %H:%M"),
                "status":"Terminé",
                "extrait":ligne[3][:100]+"..."
            })
        return historique
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Erreur base de données:{e}")