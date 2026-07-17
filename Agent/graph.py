from langgraph.graph import StateGraph, START, END
from Agent.state import AgentState
from Agent.nodes import create_nodes

def build_workflow(rag_tool):
    planificateur_node,extracteur_node,redacteur_node = create_nodes(rag_tool)

    builder=StateGraph(AgentState)

    builder.add_node("planificateur",planificateur_node)
    builder.add_node("extracteur",extracteur_node)
    builder.add_node("redacteur",redacteur_node)

    builder.add_edge(START,"planificateur")
    builder.add_edge("planificateur","extracteur")
    builder.add_edge("extracteur","redacteur")
    builder.add_edge("redacteur",END)

    return builder.compile()