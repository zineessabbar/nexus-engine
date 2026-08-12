import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agent.agent_tool import init_rag_tool
from Agent.graph import build_workflow
from logger import get_logger

logger = get_logger("agent.main")


def main():
    print("Démarrage du workflow NEXUS...")
    rag_tool = init_rag_tool()
    workflow = build_workflow(rag_tool)

    print("Tapez 'quitter' pour arrêter.\n")
    while True:
        description = input("Soumettez une description de projet : ").strip()
        if description.lower() in ("quitter", "quit", "exit", "q"):
            break
        if not description:
            continue

        print("\nDémarrage de l'audit...")
        state = {
            "description_projet": description,
            "domaines_identifies": [],
            "textes_normatifs": [],
            "rapport_final": "",
        }

        final_state = workflow.invoke(state)
        print("\n" + "=" * 60)
        print(final_state.get("rapport_final", "Aucun rapport généré."))
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()