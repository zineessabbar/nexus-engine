from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from Agent.state import AgentState
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import LLM_MODEL,LLM_TEMPERATURE
from prompts import get_prompt
from logger import get_logger

logger = get_logger("agent")

from llm_providers import LLMProvider, OllamaProvider
from config import LLM_MODEL, LLM_TEMPERATURE
from langchain_core.messages import AIMessage

SYSTEM_PROMPT_NODE = """Tu es un auditeur IT senior à la BCP.
Réponds uniquement en JSON valide, sans texte autour."""


def create_nodes(rag_tool, llm_provider: LLMProvider = None):

    # --- Initialisation du LLM ---
    if llm_provider is None:
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

        def _invoke(system: str, human: str) -> str:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=human),
            ]
            return llm.invoke(messages).content
    else:
        def _invoke(system: str, human: str) -> str:
            return llm_provider.generate(system, human)

    # --- Nœud 1 : Planificateur ---
    def planificateur(state: AgentState) -> dict:
        logger.info("Etape 1 : Identification des domaines")
        projet = state.get("description_projet", "")
        prompt = get_prompt("planificateur", projet=projet)
        content = _invoke(SYSTEM_PROMPT_NODE, prompt)
        mots_cles = [m.strip() for m in content.split(",") if m.strip()]
        logger.info(f"Domaines identifiés : {mots_cles}")
        return {"domaines_identifies": mots_cles}

    # --- Nœud 2 : Extracteur (parallèle) ---
    def extracteur(state: AgentState) -> dict:
        domaines = state.get("domaines_identifies", [])
        logger.info(f"Etape 2 : RAG sur {len(domaines)} domaines")
        textes_trouves = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    rag_tool.recherche_normes, domaine
                ): domaine
                for domaine in domaines
            }
            for future in as_completed(futures):
                domaine = futures[future]
                try:
                    resultat = future.result()
                    if resultat and "Aucune" not in resultat:
                        textes_trouves.append(
                            f"--- Normes pour '{domaine}' ---\n{resultat}"
                        )
                    else:
                        logger.debug(f"Aucune norme pour '{domaine}'")
                except Exception as e:
                    logger.error(f"Erreur RAG sur '{domaine}': {e}")

        if not textes_trouves:
            textes_trouves.append("Aucun article normatif trouvé pour ce projet.")

        return {"textes_normatifs": textes_trouves}

    # --- Nœud 3 : Rédacteur ---
    def redacteur(state: AgentState) -> dict:
        logger.info("Etape 3 : Génération du rapport")
        normes = "\n\n".join(state.get("textes_normatifs", []))
        prompt = get_prompt(
            "redacteur",
            normes=normes,
            projet=state["description_projet"],
        )
        content = _invoke(
            "Tu es un auditeur de conformité IT senior à la BCP.",
            prompt,
        )
        return {"rapport_final": content}

    return planificateur, extracteur, redacteur