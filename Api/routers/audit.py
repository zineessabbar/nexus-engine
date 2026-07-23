import uuid
from datetime import datetime
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json


from Agent.agent_tool import init_rag_tool
from Agent.graph import build_workflow
from Rag.vector_store import get_db_conn

workflow=None

def init_history_table():
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS audit_history (id SERIAL PRIMARY KEY,titre_projet TEXT,description TEXT, domaines_identifies TEXT, rapport_final TEXT,date_audit TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"verifier la table d'historique:{e}")


@asynccontextmanager
async def router_lifespan(router:APIRouter):
    global workflow
    init_history_table()
    try:
        outil_rag=init_rag_tool()
        workflow =build_workflow(outil_rag)
        print(f"Moteur IA pret")
    except Exception as e:
        print(f"Erreur los de l'initialisation du moteur:{e}")
        workflow=None
    yield

router = APIRouter(lifespan=router_lifespan)

class ProjectRequest(BaseModel):
    description:str

class AuditResponse(BaseModel):
    domaines_identifies:list[str]
    rapport_final:str

@router.post("/analyse",response_model=AuditResponse)
async def analyse_project(request:ProjectRequest):
    if not workflow:
        raise HTTPException(status_code=503,detail="LangGraph non initialisé")
    try:
        config={"configurable":{"thread_id":str(uuid.uuid4())}}
        state = {
            "description_projet":request.description,
            "domaines_identifies":[],
            "textes_normatifs":[],
            "rapport_final":""
        }
        final_state=await asyncio.to_thread(workflow.invoke,state,config)
        
        domaines=final_state.get("domaines_identifies",[])
        rapport=final_state.get("rapport_final","Erreur:Aucaun rapport généré")

        titre_court=request.description[:40]+"..." if len(request.description)> 40 else request.description

        def save_to_db():
            conn=get_db_conn()
            try:
                cursor=conn.cursor()
                cursor.execute("""INSERT INTO audit_history(titre_projet ,description , domaines_identifies,rapport_final)VALUES(%s,%s,%s,%s)""",(titre_court, request.description,json.dumps(domaines),rapport))
                conn.commit()
                cursor.close()
            finally:
                conn.close()
        await asyncio.to_thread(save_to_db)
        return AuditResponse(
            domaines_identifies=domaines,
            rapport_final=rapport
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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