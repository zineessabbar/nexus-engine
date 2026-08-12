# Moteur d'Architecture et de Conformité IT (RAG + LangGraph)

## Architecture
Documents normatifs (5 chartes)
↓
Chunking structure-aware → BGE-M3 embeddings → pgvector
↓
Retrieval hybride : vectoriel + BM25 + RRF + BGE-reranker
↓
Agent LangGraph : planificateur → extracteur → rédacteur
↓
FastAPI REST : 
 - POST /audit/analyse (Standard)
 - POST /audit/analyse/stream (Server-Sent Events)
 - GET /system/history | GET /system/stats
↓
Rapport de Conformité structuré

---

## Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | BAAI/bge-m3 (local) |
| Reranking | BAAI/bge-reranker-v2-m3 |
| LLM (Hybride) | Qwen 2.5:7B (Ollama Local) / GPT-4o & GPT-4o-Mini (OpenAI Cloud) |
| Base vectorielle | PostgreSQL + pgvector |
| Recherche lexicale | BM25 (rank-bm25) |
| Agent | LangGraph |
| API | FastAPI |

---

## Utilisation

### Via API (Postman ou Frontend)
**Endpoint Standard :** `POST http://localhost:8000/api/v1/audit/analyse`
**Endpoint Streaming :** `POST http://localhost:8000/api/v1/audit/analyse/stream`

Content-Type: application/json
```json
{
  "description": "Application web bancaire avec authentification simple et données hébergées sur AWS S3...",
  "model": "qwen2.5:7b" 
}

Via terminal
python -m Agent.main

Évaluation Automatisée (Ragas)
Bash
python -m Test.ragas_eval
(Évalue la fidélité, la pertinence de la réponse et la précision du contexte via un LLM "Juge").

Structure du projet
├── config.py                  # Configuration centralisée et modèles disponibles
├── logger.py                  # Logs professionnels
├── prompts.yaml               # Prompts LLM externalisés
├── prompts.py                 # Chargeur de prompts
├── llm_providers.py           # Routage dynamique des LLMs (Ollama / OpenAI)
├── docker-compose.yml         # Conteneur pour PostgreSQL + pgvector
├── requirements.txt
├── .env.example
├── Rag/
│   ├── chunking.py            # Découpage structure-aware
│   ├── document_loader.py     # Extraction OCR, PDF, DOCX
│   ├── embeddings.py          # Génération embeddings BGE-M3
│   ├── retrieval.py           # Retrieval hybride + reranking
│   ├── vector_store.py        # Indexation pgvector (batch)
│   ├── generation.py          # Génération avec citation
│   └── documents/             # Chartes normatives BCP (5 fichiers)
├── Agent/
│   ├── state.py               # État LangGraph
│   ├── nodes.py               # Nœuds : planificateur, extracteur, rédacteur
│   ├── graph.py               # Graphe LangGraph
│   ├── rag_tool.py            # RAG encapsulé comme outil
│   ├── agent_tool.py          # Initialisation des ressources
│   └── main.py                # Interface terminal
├── Api/
│   ├── main.py                # Application FastAPI
│   └── routers/
│       ├── audit.py           # Routes d'audit (Standard & Stream)
│       └── system.py          # GET /health, GET /stats, POST /reload
└── Test/
├── test.py                # Script de validation
├── ragas_eval.py          # Pipeline d'évaluation Ragas
└── rapport_validation.json