"""
Évaluation RAGAS — Projet NEXUS
Mesure la qualité du pipeline RAG sur 3 métriques standards :
- Faithfulness : la réponse est-elle ancrée sur le contexte fourni ?
- Answer Relevancy : la réponse répond-elle à la question posée ?
- Context Precision : les chunks récupérés sont-ils pertinents ?
"""

import sys
from unittest.mock import MagicMock
# On fait croire à Python que le module existe pour calmer Ragas
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

# --- Le reste de ton code normal en dessous ---
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragas.run_config import RunConfig
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from Agent.agent_tool import init_rag_tool
from Rag.retrieval import hybrid_search

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import LLM_MODEL

# ---------------------------------------------------------------------------
# JEU D'ÉVALUATION
# ---------------------------------------------------------------------------

EVAL_SET = [
    {
        "question": "Quelle est la longueur minimale pour les mots de passe ?",
        "ground_truth": "Les mots de passe doivent avoir une longueur minimale de douze caractères.",
    },
    {
        "question": "Quel algorithme de chiffrement est obligatoire pour les données sensibles ?",
        "ground_truth": "Les données sensibles doivent être chiffrées selon l'algorithme AES-256.",
    },
    {
        "question": "Peut-on héberger des données de niveau 1 sur un cloud public ?",
        "ground_truth": "Non, les données de niveau 1 doivent être hébergées exclusivement sur l'infrastructure interne ou un cloud privé certifié sur le territoire national.",
    },
    {
        "question": "Que faut-il faire en cas d'exception à l'obligation de chiffrement ?",
        "ground_truth": "Une dérogation écrite signée par le RSSI est requise, avec une justification documentée et une durée de validité de six mois renouvelables.",
    },
    {
        "question": "Quelles sont les règles pour les APIs exposées à des tiers ?",
        "ground_truth": "Toute API exposée à des tiers doit implémenter une authentification OAuth 2.0 et un mécanisme de rate limiting.",
    },
]

# ---------------------------------------------------------------------------
# PIPELINE D'ÉVALUATION
# ---------------------------------------------------------------------------

def build_eval_dataset(rag_tool) -> Dataset:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in EVAL_SET:
        question = item["question"]
        
        # Retrieval
        conn = rag_tool.conn if hasattr(rag_tool, 'conn') else None
        from Rag.vector_store import get_db_conn
        conn = get_db_conn()
        
        try:
            chunks = hybrid_search(
                query=question,
                conn=conn,
                embedding_model=rag_tool.embedding_model,
                bm25_index=rag_tool.bm25_index,
                all_chunks=rag_tool.all_chunks,
                chunks_by_id=rag_tool.chunks_by_id,
                cross_encoder=rag_tool.cross_encoder,
            )
        finally:
            conn.close()

        # Génération
        result = rag_tool.recherche_normes(question)
        
        questions.append(question)
        answers.append(result)
        contexts.append([c["text"] for c in chunks])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    print("Chargement du RAG tool...")
    rag_tool = init_rag_tool()

    print("Construction du dataset d'évaluation...")
    dataset = build_eval_dataset(rag_tool)

    print("Configuration du Juge LLM local (Ollama)...")
    
    # 1. Le LLM Juge (Température à 0 pour des évaluations strictes et constantes)
    judge_llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
    
    # 2. Le modèle d'Embedding Juge (Requis par Ragas pour Answer Relevancy)
    # Remplacer "BAAI/bge-m3" par le nom exact du modèle défini dans ton config.py
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    print("Lancement de l'évaluation RAGAS...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=300, max_workers=1)  # On bride Ragas pour protéger Ollama
    )

    print("\n" + "=" * 60)
    print("RÉSULTATS RAGAS — PROJET NEXUS")
    print("=" * 60)
    print(f"Faithfulness      : {results['faithfulness']:.3f}  (réponse ancrée sur le contexte)")
    print(f"Answer Relevancy  : {results['answer_relevancy']:.3f}  (réponse pertinente)")
    print(f"Context Precision : {results['context_precision']:.3f}  (chunks pertinents récupérés)")
    print("=" * 60)

    # Sauvegarde
    import json
    os.makedirs("Test", exist_ok=True)
    with open("Test/ragas_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "faithfulness": results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"],
            "context_precision": results["context_precision"],
        }, f, ensure_ascii=False, indent=2)
    print("Résultats sauvegardés : Test/ragas_results.json")


if __name__ == "__main__":
    main()