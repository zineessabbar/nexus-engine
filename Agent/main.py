from Agent.agent_tool import init_rag_tool
from Agent.graph import build_workflow
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rag')))
from langchain_core.messages import HumanMessage

def main():
    print("démarrage du workflow")
    try:
        outil_rag=init_rag_tool()
        workflow=build_workflow(outil_rag)
    except Exception as e:
        print(f"Erreur de chargement des outils: {e}")
        sys.exit(1)
    print(" tapez 'quitter' ou 'quit' ou 'exit' ou 'q' pour quitter")

    config={"configurable":{"thread_id":"session_architecture_1"}}    
    while True:
        user_input=input("Soumettez une description projet ou une question : ")
        if user_input.lower() in ["quitter", "quit","exit" , "q"]:
            break
        state={
            "description_projet":user_input,
            "domaines_identifies":[],
            "textes_normatifs":[],
            "rapport_final":""
        }

        print("Démarrage de l'audit")
        final_state=workflow.invoke(state,config)

        print(final_state.get("rapport_final","Erreur:Aucun rapport généré"))

if __name__ == "__main__":
    main()