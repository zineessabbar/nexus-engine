from retrieval import hybrid_search
from abc import ABC, abstractmethod

NOT_FOUND_MESSAGE="Je n'ai trouvé aucune information dans les documents fournis."


SYSTEM_PROMPT = """Tu es un assistant qui répond à des questions sur les normes et chartes internes d'une banque, en te basant EXCLUSIVEMENT sur les extraits de documents fournis dans le contexte ci-dessous.

RÈGLES STRICTES :
1. Ne réponds qu'à partir des extraits fournis. N'utilise aucune connaissance générale sur les banques, la sécurité informatique, ou la réglementation, même si elle te semble correcte.
2. Pour CHAQUE affirmation factuelle, cite précisément sa source au format suivant : (Source : [nom du document], Article [numéro]).
3. Tu PEUX et DOIS faire des déductions logiques directes à partir du texte des extraits fournis, même si la question n'utilise pas exactement les mêmes mots que le texte source. Par exemple, si un extrait dit qu'une ressource doit être "exclusivement" hébergée en interne, tu peux en déduire que l'hébergement sur un service externe n'est pas autorisé, et le dire explicitement en citant cet extrait. Cette règle ne t'autorise PAS à ajouter une information qui ne découle pas directement du texte fourni — uniquement à reformuler ou interpréter ce que dit déjà l'extrait.
4. Si les extraits fournis ne permettent pas de répondre à la question, MÊME PAR DÉDUCTION DIRECTE selon la règle 3, réponds explicitement : "Je ne trouve pas d'information à ce sujet dans les documents fournis." Ne complète JAMAIS une réponse partielle en devinant le reste.
5. Si plusieurs extraits semblent se contredire, signale-le explicitement plutôt que de choisir silencieusement l'un des deux.
6. Reste concis et factuel. N'ajoute pas de recommandations ou d'opinions qui ne sont pas explicitement écrites dans les extraits."""

class LLMProvider(ABC):
    @abstractmethod
    def generate(self,system_prompt,user_prompt):
        pass

def answer_question(question,conn,embedding_model,cross_encoder,bm25_index,all_chunks,chunks_by_id,llm_provider):
    top_chunks = hybrid_search(
        query=question, conn=conn, embedding_model=embedding_model, bm25_index=bm25_index, all_chunks=all_chunks, chunks_by_id=chunks_by_id, cross_encoder=cross_encoder
    )
    top_score = top_chunks[0].get("rerank_score", 0.0) if top_chunks else 0.0
    if should_skip(top_chunks,top_relevance_score=top_score):
        return {
            "question":question,
            "answer":NOT_FOUND_MESSAGE,
            "source_used":[],
            "generation_skipped":True,
            }
    filtered_chunks = [
        chunk for chunk in top_chunks 
        if chunk.get("rerank_score", -999.0) >= RERANK_SCORE_THRESHOLD
    ]

    user_prompt=build_user_prompt(question,filtered_chunks)
    answer=llm_provider.generate(SYSTEM_PROMPT,user_prompt)
    
    return {
        "question":question,
        "answer":answer,
        "source_used":[(c['doc_source'],c['article_number']) for c in filtered_chunks],
        "generation_skipped":False,
    }

class OllamaProvider(LLMProvider):
    def __init__(self,model_name="qwen2.5:7b"):
        self.model_name = model_name

    def generate(self,system_prompt,user_prompt):
        # pyrefly: ignore [missing-import]
        import ollama
        response = ollama.chat(model=self.model_name,messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],)
        return response['message']['content']

def format_context(chunks):
    if not chunks:
        return "Aucune extrait pertinent trouvé dans le corpus."
    blocks=[]
    for i , chunk in enumerate(chunks,1):
        source_label=f"{chunk['doc_source']},Article {chunk['article_number']}"
        blocks.append(f"[Extrait {i}] (Source:{source_label})\n{chunk['text']}")
    return "\n\n".join(blocks)

def build_user_prompt(question,chunks):
    context=format_context(chunks)
    return f"""voici les extraits de documents pertinents trouvés dans le corpus: 
    {context} 
    -------
    Question:{question}"""

RERANK_SCORE_THRESHOLD=0.0

def should_skip(chunks,top_relevance_score):
    if not chunks:
        return True
    if top_relevance_score < RERANK_SCORE_THRESHOLD:
        return True
    return False
        
def main():
    from vector_store import get_db_conn
    from retrieval import load_all_chunks, BM25SearchIndex
    from embeddings import load_embedding_model
    import torch
    from sentence_transformers import CrossEncoder

    print(" Connexion à la base de données...")
    connection = get_db_conn()
    
    print(" Chargement des ressources en mémoire...")
    all_chunks = load_all_chunks(connection)
    chunks_by_id = {c['id']: c for c in all_chunks}
    bm25_index = BM25SearchIndex(all_chunks)
    embedding_model = load_embedding_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
    
    llm = OllamaProvider(model_name="qwen2.5:7b")
    
    query = "on peut heberger des données sensibles sur un cloud public?"
    print(f"\n Question posée : {query}")
    
    result = answer_question(
        query, connection, embedding_model, cross_encoder, bm25_index, all_chunks, chunks_by_id, llm
    )
    
    print("\n Réponse finale du RAG :")
    print(result["answer"])
    print(f"\nSources utilisées : {result['source_used']}")
    
    connection.close()

if __name__ == "__main__":
    main()