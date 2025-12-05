from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from helper_agent.agent.offline.nodes import (
    create_evaluate_node,
    create_generate_node,
    create_reformulate_node,
    create_retrieve_node,
)
from helper_agent.agent.offline.state import OfflineAgentState
from helper_agent.utilities.llm import get_llm
from helper_agent.utilities.logger import get_logger
from helper_agent.vectorstore.retriever import Retriever

logger = get_logger("helper_agent")


def _should_continue_factory(
    max_retries: int,
) -> Callable[[OfflineAgentState], Literal["end", "reformulate"]]:
    """
    Build a continuation callback bound to the configured retry budget.

    :param max_retries: Maximum number of reformulation attempts
    :return: Callback deciding whether to end or reformulate
    """

    def _should_continue(state: OfflineAgentState) -> Literal["end", "reformulate"]:
        if state.get("is_confident", False):
            logger.debug("Answer is confident, ending.")
            return "end"

        retry_count = state.get("retry_count", 0)

        if retry_count >= max_retries:
            logger.debug(f"Max retries ({max_retries}) reached, ending.")
            return "end"

        logger.debug(
            f"Not confident, reformulating (retry {retry_count}/{max_retries})."
        )
        return "reformulate"

    return _should_continue


def build_offline_graph(
    prompts_dir: str,
    llm_config: dict[str, Any],
    vectordb_path: str,
    collection_name: str,
    top_k: int = 5,
    max_retries: int = 2,
) -> StateGraph[OfflineAgentState]:
    """
    Build the offline LangGraph agent graph.

    :param prompts_dir: Directory containing prompt templates
    :param llm_config: LLM defaults/overrides used by get_llm
    :param vectordb_path: Path to persisted vector database
    :param collection_name: Vector collection to search
    :param top_k: Number of context chunks to retrieve per query
    :param max_retries: Max reformulation attempts before stopping
    :return: Compiled StateGraph
    """
    retriever = Retriever(
        vectordb_path=vectordb_path,
        collection_name=collection_name,
    )

    # init LLMs
    generate_llm = get_llm("generate", llm_config)
    evaluate_llm = get_llm("evaluate", llm_config)
    reformulate_llm = get_llm("reformulate", llm_config)

    # node functions
    retrieve_node = create_retrieve_node(retriever, top_k)
    generate_node = create_generate_node(generate_llm, prompts_dir)
    evaluate_node = create_evaluate_node(evaluate_llm, prompts_dir)
    reformulate_node = create_reformulate_node(reformulate_llm, prompts_dir)

    # build graph
    graph = StateGraph(OfflineAgentState)

    # add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("reformulate", reformulate_node)

    # add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    # conditional edge
    graph.add_conditional_edges(
        "evaluate",
        _should_continue_factory(max_retries),
        {
            "end": END,
            "reformulate": "reformulate",
        },
    )

    # reformulate loops back to retrieve
    graph.add_edge("reformulate", "retrieve")

    return graph.compile()


def run_offline_agent(
    graph: StateGraph[OfflineAgentState], query: str
) -> OfflineAgentState:
    """
    Run the agent with a query.

    :param graph: Compiled StateGraph
    :param query: User query
    :return: Final state
    """
    initial_state = {
        "query": query,
        "reformulated_query": "",
        "retrieved_docs": [],
        "answer": "",
        "is_confident": False,
        "retry_count": 0,
    }

    final_state = graph.invoke(initial_state)
    return final_state
