## Architecture
Documents normatifs (5 chartes)
↓
Chunking structure-aware → BGE-M3 embeddings → pgvector
↓
Retrieval hybride : vectoriel + BM25 + RRF + BGE-reranker
↓
Agent LangGraph : planificateur → extracteur → rédacteur
↓
FastAPI REST : POST /audit | GET /history | GET /stats
↓
Rapport de Conformité structuré

---

## Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | BAAI/bge-m3 (local) |
| Reranking | BAAI/bge-reranker-v2-m3 |
| LLM | qwen2.5:7b-instruct via Ollama |
| Base vectorielle | PostgreSQL + pgvector |
| Recherche lexicale | BM25 (rank-bm25) |
| Agent | LangGraph |
| API | FastAPI |

---

## Utilisation

### Via API (Postman)
POST http://localhost:8000/api/v1/audit/analyse
Content-Type: application/json
{
"description": "Application web bancaire avec authentification
simple et données hébergées sur AWS S3..."
}

### Via terminal

```bash
python -m Agent.main
```

---

## Tests de validation

```bash
python -m Test.test
```

Résultat obtenu : **11/11 tests passés (100%)**

| Catégorie | Résultat |
|---|---|
| Tests RAG (pertinence retrieval) | 5/5 ✅ |
| Tests Agent (détection non-conformités) | 3/3 ✅ |
| Tests hors-corpus (anti-hallucination) | 3/3 ✅ |

---

## Structure du projet
├── config.py                  # Configuration centralisée
├── logger.py                  # Logs professionnels
├── prompts.yaml               # Prompts LLM externalisés
├── prompts.py                 # Chargeur de prompts
├── requirements.txt
├── .env.example
├── Rag/
│   ├── chunking.py            # Découpage structure-aware
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
│       ├── audit.py           # POST /analyse, GET /history
│       └── system.py          # GET /health, GET /stats
└── Test/
├── test.py                # Script de validation
└── rapport_validation.json

---
