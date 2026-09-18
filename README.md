# Projet NEXUS : Moteur de Conformité Architecturale IA (Engine)
###  Note de Confidentialité : 
Ce dépôt contient une version anonymisée (Proof of Concept) du moteur d'évaluation de conformité architecturale conçu lors d'un stage au sein de la Direction Architecture des Systèmes d'Information de la Banque Centrale Populaire (BCP). Les chartes internes, les données normatives et l'infrastructure propriétaire ont été retirées ou remplacées par des données de test.

###  Interface Client : 
Ce dépôt contient uniquement le backend IA. L'interface utilisateur correspondante, développée avec React 18, Vite et Zustand, est disponible dans le dépôt [nexus-ui](https://github.com/zineessabbar/nexus-ui)

## Présentation
NEXUS est un moteur d'audit automatisé combinant un pipeline RAG (Retrieval-Augmented Generation) et un agent d'orchestration LangGraph pour confronter automatiquement les architectures des projets IT aux normes de sécurité, de données et d'hébergement. Conçu pour le secteur bancaire, le système garantit une stricte souveraineté des données en exécutant l'intégralité de la chaîne d'intelligence artificielle localement.

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


## Stack technique
Orchestration IA : LangGraph

Modèles de Langage : Qwen 2.5 7B via Ollama (Local)- Abstraction multi-provider pour GPT-4o

Modèles d'Embedding : BAAI/bge-m3 et BAAI/bge-reranker-v2-m3

Base de Données Vectorielle : PostgreSQL avec l'extension pgvector

API & Streaming : FastAPI avec implémentation du Server-Sent Events (SSE) pour le streaming de la réponse token par token


## Utilisation
Le système expose plusieurs endpoints, dont POST /api/v1/audit/analyse pour soumettre les descriptions d'architecture.

```bash

#Lancement du serveur FastAPI en local
python -m Api.main

#Exécution de la suite de tests formels
python -m Test.test
```

