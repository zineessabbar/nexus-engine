import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, START, END
from Agent.state import AgentState
from Agent.nodes import create_nodes
from llm_providers import LLMProvider

def build_workflow_with_provider(rag_tool, llm_provider=None):
    planificateur, extracteur, redacteur = create_nodes(
        rag_tool, llm_provider=llm_provider
    )
    builder = StateGraph(AgentState)
    builder.add_node("planificateur", planificateur)
    builder.add_node("extracteur", extracteur)
    builder.add_node("redacteur", redacteur)
    builder.add_edge(START, "planificateur")
    builder.add_edge("planificateur", "extracteur")
    builder.add_edge("extracteur", "redacteur")
    builder.add_edge("redacteur", END)
    return builder.compile()