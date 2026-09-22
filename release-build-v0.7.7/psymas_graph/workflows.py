"""LangGraph assembly functions.

Node implementations are injected so this module owns graph topology without
depending on the large detector implementation module.
"""

from collections.abc import Callable, Mapping, Sequence

from langgraph.graph import END, StateGraph

from .state import State


FORENSIC_SPECIALIST_AGENT_ORDER: tuple[str, ...] = (
    "nm_agent",
    "pm_agent",
    "ac_agent",
    "as_agent",
    "pk_agent",
    "rg_agent",
    "cp_agent",
    "tt_agent",
)


def build_forensic_workflow(
    *,
    router: Callable,
    specialists: Mapping[str, Callable],
    synthesizer: Callable,
    reporter: Callable,
    specialist_order: Sequence[str] = FORENSIC_SPECIALIST_AGENT_ORDER,
):
    """Build the parallel deterministic-evidence workflow."""
    workflow = StateGraph(State)
    workflow.add_node("router", router)
    for agent_name in specialist_order:
        workflow.add_node(agent_name, specialists[agent_name])
    workflow.add_node("synthesizer", synthesizer)
    workflow.add_node("reporter", reporter)
    workflow.set_entry_point("router")
    for agent_name in specialist_order:
        workflow.add_edge("router", agent_name)
        workflow.add_edge(agent_name, "synthesizer")
    workflow.add_edge("synthesizer", "reporter")
    workflow.add_edge("reporter", END)
    return workflow.compile()


def build_psychometric_workflow(
    *,
    orchestrator: Callable,
    irt: Callable,
    response_time: Callable,
    analyzer: Callable,
):
    """Build the response/RT preparation and IRT workflow."""
    workflow = StateGraph(State)
    workflow.add_node("Orchestrator_node", orchestrator)
    workflow.add_node("irt_node", irt)
    workflow.add_node("rt_node", response_time)
    workflow.add_node("Analyze_node", analyzer)
    workflow.set_entry_point("Orchestrator_node")
    workflow.add_edge("Orchestrator_node", "irt_node")
    workflow.add_edge("Orchestrator_node", "rt_node")
    workflow.add_edge("irt_node", "Analyze_node")
    workflow.add_edge("rt_node", "Analyze_node")
    workflow.add_edge("Analyze_node", END)
    return workflow.compile()
