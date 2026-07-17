"""
Script de validation
Semaine 8 : Tests & Validation
"""

import json
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agent.agent_tool import init_rag_tool
from Agent.graph import build_workflow


# ---------------------------------------------------------------------------
# CAS DE TEST
# ---------------------------------------------------------------------------

TESTS_RAG = [
    {
        "id": "RAG-01",
        "question": "Quelle est la longueur minimale pour les mots de passe ?",
        "article_attendu": "1.2",
    },
    {
        "id": "RAG-02",
        "question": "Quel algorithme de chiffrement est obligatoire pour les données sensibles ?",
        "article_attendu": "2.1",
    },
    {
        "id": "RAG-03",
        "question": "Qu'est-ce que dit la norme sur l'authentification à deux facteurs ?",
        "article_attendu": "1.1",
    },
    {
        "id": "RAG-04",
        "question": "Quelles sont les règles pour héberger des données critiques de niveau 1 ?",
        "article_attendu": "1.1",
    },
    {
        "id": "RAG-05",
        "question": "Que faut-il faire en cas d'exception à l'obligation de chiffrement ?",
        "article_attendu": "2.3",
    },
]

TESTS_AGENT = [
    {
        "id": "AGENT-01",
        "description_projet": (
            "Nouvelle application mobile bancaire. "
            "Authentification par code PIN à 4 chiffres uniquement, pas de 2FA. "
            "Données utilisateurs stockées en clair sur serveur AWS. "
            "Pas de chiffrement en transit."
        ),
        "non_conformites_attendues": ["1.1", "2.1", "2.2", "3.1"],
    },
    {
        "id": "AGENT-02",
        "description_projet": (
            "Migration de la base de données clients vers Azure Cloud public. "
            "Les données incluent numéros de compte, soldes et transactions. "
            "Aucune analyse de réversibilité prévue."
        ),
        "non_conformites_attendues": ["1.1", "2.1"],
    },
    {
        "id": "AGENT-03",
        "description_projet": (
            "Nouveau flux de données financières vers un prestataire externe. "
            "Les données transitent sans chiffrement. "
            "Aucune validation par le comité architecture prévue. "
            "Logs envoyés vers un SaaS externe non référencé."
        ),
        "non_conformites_attendues": ["2.1", "2.2", "2.3"],
    },
]

TESTS_HORS_CORPUS = [
    {"id": "HC-01", "question": "Quelle est la recette de la pastilla ?"},
    {"id": "HC-02", "question": "Quel est le cours actuel du bitcoin ?"},
    {"id": "HC-03", "question": "Comment configurer un routeur Cisco ?"},
]

SIGNAL_NON_TROUVE = "Aucune norme applicable n'a été trouvée"


# ---------------------------------------------------------------------------
# FONCTIONS DE TEST
# ---------------------------------------------------------------------------

def test_rag(rag_tool, cas):
    """Vérifie que le RAG retrouve l'article attendu pour une question."""
    resultat = rag_tool.recherche_normes(cas["question"])
    article_trouve = f"Article {cas['article_attendu']}" in resultat
    return {
        "id": cas["id"],
        "question": cas["question"],
        "article_attendu": cas["article_attendu"],
        "succes": article_trouve,
        "details": "Article trouvé" if article_trouve else f"Article {cas['article_attendu']} absent du résultat",
    }


def test_agent(workflow, cas):
    """Vérifie que l'agent détecte les non-conformités attendues."""
    state = {
        "description_projet": cas["description_projet"],
        "domaines_identifies": [],
        "textes_normatifs": [],
        "rapport_final": "",
    }
    final_state = workflow.invoke(state, {"configurable": {"thread_id": cas["id"]}})
    rapport = final_state.get("rapport_final", "")

    articles_detectes = []
    articles_manques = []
    for article in cas["non_conformites_attendues"]:
        if f"Article {article}" in rapport or article in rapport:
            articles_detectes.append(article)
        else:
            articles_manques.append(article)

    succes = len(articles_manques) == 0
    return {
        "id": cas["id"],
        "articles_attendus": cas["non_conformites_attendues"],
        "articles_detectes": articles_detectes,
        "articles_manques": articles_manques,
        "succes": succes,
        "taux_detection": f"{len(articles_detectes)}/{len(cas['non_conformites_attendues'])}",
    }


def test_hors_corpus(rag_tool, cas):
    """Vérifie que le système répond 'non trouvé' sur une question hors-corpus."""
    resultat = rag_tool.recherche_normes(cas["question"])
    est_non_trouve = (
        "Aucune information pertinente" in resultat
        or "Aucune norme" in resultat
    )
    return {
        "id": cas["id"],
        "question": cas["question"],
        "succes": est_non_trouve,
        "details": "Refus correct" if est_non_trouve else "HALLUCINATION DETECTEE",
        "reponse_brute": resultat[:150] + "..." if len(resultat) > 150 else resultat,
    }


# ---------------------------------------------------------------------------
# RAPPORT DE VALIDATION
# ---------------------------------------------------------------------------

def afficher_rapport(resultats_rag, resultats_agent, resultats_hc):
    separateur = "=" * 65

    print(f"\n{separateur}")
    print("RAPPORT DE VALIDATION ")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(separateur)

    # --- RAG ---
    print("\n[1] TESTS RAG — Pertinence du retrieval")
    print("-" * 65)
    succes_rag = sum(1 for r in resultats_rag if r["succes"])
    for r in resultats_rag:
        statut = "✅ PASS" if r["succes"] else "❌ FAIL"
        print(f"  {r['id']} | {statut} | Article attendu : {r['article_attendu']} | {r['details']}")
    print(f"\n  Résultat : {succes_rag}/{len(resultats_rag)} tests passés")

    # --- AGENT ---
    print("\n[2] TESTS AGENT — Détection des non-conformités")
    print("-" * 65)
    succes_agent = sum(1 for r in resultats_agent if r["succes"])
    for r in resultats_agent:
        statut = "✅ PASS" if r["succes"] else "❌ FAIL"
        print(f"  {r['id']} | {statut} | Détection : {r['taux_detection']}")
        if r["articles_manques"]:
            print(f"         Articles manqués : {r['articles_manques']}")
    print(f"\n  Résultat : {succes_agent}/{len(resultats_agent)} tests passés")

    # --- HORS CORPUS ---
    print("\n[3] TESTS HORS-CORPUS — Anti-hallucination")
    print("-" * 65)
    succes_hc = sum(1 for r in resultats_hc if r["succes"])
    for r in resultats_hc:
        statut = "✅ PASS" if r["succes"] else "❌ FAIL"
        print(f"  {r['id']} | {statut} | {r['details']}")
        if not r["succes"]:
            print(f"         Réponse : {r['reponse_brute']}")
    print(f"\n  Résultat : {succes_hc}/{len(resultats_hc)} tests passés")

    # --- SYNTHESE ---
    total = len(resultats_rag) + len(resultats_agent) + len(resultats_hc)
    total_succes = succes_rag + succes_agent + succes_hc
    taux = round(total_succes / total * 100, 1)

    print(f"\n{separateur}")
    print(f"SYNTHESE GLOBALE : {total_succes}/{total} tests passés ({taux}%)")

    if taux == 100:
        print("Statut : VALIDATION COMPLÈTE ✅")
    elif taux >= 80:
        print("Statut : VALIDATION PARTIELLE ⚠️  — points à corriger avant livraison")
    else:
        print("Statut : ÉCHEC ❌ — révision nécessaire")
    print(separateur)

    # Sauvegarde JSON
    rapport = {
        "date": datetime.now().isoformat(),
        "taux_global": taux,
        "tests_rag": resultats_rag,
        "tests_agent": resultats_agent,
        "tests_hors_corpus": resultats_hc,
    }
    os.makedirs("Test", exist_ok=True)
    with open("Test/rapport_validation.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    print(f"\nRapport sauvegardé : Test/rapport_validation.json")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Chargement des ressources...")
    rag_tool = init_rag_tool()
    workflow = build_workflow(rag_tool)
    print("Ressources chargées.\n")

    print("Lancement des tests RAG...")
    resultats_rag = [test_rag(rag_tool, cas) for cas in TESTS_RAG]

    print("Lancement des tests agent (plus long — LLM impliqué)...")
    resultats_agent = [test_agent(workflow, cas) for cas in TESTS_AGENT]

    print("Lancement des tests hors-corpus...")
    resultats_hc = [test_hors_corpus(rag_tool, cas) for cas in TESTS_HORS_CORPUS]

    afficher_rapport(resultats_rag, resultats_agent, resultats_hc)


if __name__ == "__main__":
    main()