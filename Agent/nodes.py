from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from Agent.state import AgentState
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import LLM_MODEL,LLM_TEMPERATURE
from prompts import get_prompt
from logger import get_logger

logger = get_logger("agent")

def create_nodes(rag_tool):
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    def planificateur(state: AgentState):
        logger.info("Etape 1:Identification des domaines")
        projet = state.get('description_projet', 'Aucune description fournie')
        prompt=get_prompt("planificateur",projet=projet)
        reponse = llm.invoke([HumanMessage(content=prompt)])
        mots_cles = [mot.strip() for mot in reponse.content.split(',') if mot.strip()]
        logger.info(f"Domaines identifiés: {mots_cles}")
        return {"domaines_identifies": mots_cles}


    def extracteur(state: AgentState):
        domaines = state.get("domaines_identifies",[])
        logger.info(f"Lancement du RAG sur les domaines : {domaines}")
        textes_trouves = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures={
                executor.submit(rag_tool.recherche_normes,domaine):domaine
                for domaine in domaines
            }
            for future in as_completed(futures):
                domaine=futures[future]
                try:
                    resultat = future.result()

                    if "Aucune information pertinente" not in resultat and "Aucune norme" not in resultat:
                        textes_trouves.append(f"--- Normes pour '{domaine}': \n{resultat}")
                except Exception as e:
                    logger.error(f"Erreur sur {domaine}: {e}")

        if not textes_trouves:
            textes_trouves.append("Aucun article normatif trouvé pour ce projet.")

        return {"textes_normatifs": textes_trouves}


    def redacteur(state: AgentState):
        logger.info("Etape 3:Generation du rapport")
        toutes_les_normes = "\n\n".join(state["textes_normatifs"])
        prompt=get_prompt("redacteur",normes=toutes_les_normes,projet=state["description_projet"])
        reponse = llm.invoke([HumanMessage(content=prompt)])
        return {"rapport_final": reponse.content}

    return planificateur, extracteur, redacteur