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

workflow = None
logger = get_logger("api.audit")


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------
def init_tables():
    """Create the sessions + messages tables if they don't exist."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        # Sessions table — one row per chat thread
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Messages table — each audit exchange linked to a session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Tables sessions/messages initialisées")
    except Exception as e:
        logger.error(f"Erreur création tables : {e}")


def _persist_exchange(session_id: str, session_name: str, user_msg: str, agent_msg: str):
    """Persist a full user→agent exchange inside a session."""
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # Upsert session (create if new, ignore if exists)
        cur.execute(
            """
            INSERT INTO sessions (session_id, session_name, created_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (session_id) DO NOTHING
            """,
            (session_id, session_name),
        )

        # Insert user message
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, "user", user_msg),
        )

        # Insert agent message
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, "agent", agent_msg),
        )

        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Erreur persistance session : {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Router lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def router_lifespan(router: APIRouter):
    global workflow
    init_tables()
    try:
        outil_rag = init_rag_tool()
        workflow = build_workflow(outil_rag)
        logger.info("Workflow LangGraph initialisé")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du moteur : {e}")
        workflow = None
    yield


router = APIRouter(lifespan=router_lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class AuditResponse(BaseModel):
    domaines_identifies: list[str]
    rapport_final: str


class ProjectRequest(BaseModel):
    description: str
    model: str = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# POST /analyse  (standard, non-streaming)
# ---------------------------------------------------------------------------
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

        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
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

        rapport = final_state.get("rapport_final", "")
        domaines = final_state.get("domaines_identifies", [])

        # Persist as a session
        session_name = request.description[:80]
        _persist_exchange(session_id, session_name, request.description, rapport)

        return AuditResponse(
            domaines_identifies=domaines,
            rapport_final=rapport,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur analyse : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /analyse/stream  (SSE streaming)
# ---------------------------------------------------------------------------
@router.post("/analyse/stream")
async def analyse_project_stream(request: ProjectRequest):
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dynamic_workflow = build_workflow_with_provider(get_rag_tool(), llm_provider)

    async def generate():
        try:
            session_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": session_id}}
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

            # Persist as a session
            session_name = request.description[:80]
            _persist_exchange(session_id, session_name, request.description, rapport)

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


# ---------------------------------------------------------------------------
# GET /history  → returns SessionEntry[] for the frontend
# ---------------------------------------------------------------------------
@router.get("/history")
async def get_sessions():
    """Return all sessions in the shape the frontend expects:
    [{ session_id, session_name, created_at (unix) }]
    """
    try:
        conn = get_db_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, session_name, created_at
                FROM sessions
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()

        return [
            {
                "session_id": row[0],
                "session_name": row[1],
                "created_at": int(row[2].timestamp()) if row[2] else 0,
            }
            for row in rows
        ]

    except Exception as e:
        logger.error(f"Erreur chargement sessions : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur base de données : {e}")