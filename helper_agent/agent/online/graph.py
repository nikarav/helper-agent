from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

from helper_agent.agent.online.state import OnlineAgentState
from helper_agent.agent.online.tools import create_online_tools
from helper_agent.utilities.llm import get_llm
from helper_agent.utilities.logger import get_logger
from helper_agent.utilities.utils import load_prompt
from helper_agent.vectorstore.retriever import Retriever

logger = get_logger("helper_agent")


def _should_continue(state: OnlineAgentState) -> str:
    """
    Determine whether to continue with tools or end.

    :param state: Current agent state
    :return: Next node ("tools" or "end")
    """
    messages = state["messages"]
    last_message = messages[-1]

    # whether llm made a tool call
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.debug(
            f"Tool calls detected: {[tc['name'] for tc in last_message.tool_calls]}"
        )
        return "tools"

    logger.debug("No tool calls, ending.")
    return "end"


def build_online_graph(
    prompts_dir: str,
    llm_config: dict[str, Any],
    vectordb_path: str,
    collection_name: str,
    tavily_api_key: str,
    top_k: int = 5,
    web_search_max_results: int = 5,
) -> StateGraph[OnlineAgentState]:
    """
    Build the online ReAct agent graph.

    :param prompts_dir: Directory containing prompt templates
    :param llm_config: LLM configuration
    :param vectordb_path: Path to vector database
    :param collection_name: Vector collection name
    :param tavily_api_key: Tavily API key for web search
    :param top_k: Number of docs to retrieve
    :param web_search_max_results: Max web search results
    :return: Compiled StateGraph
    """
    # init retriever
    retriever = Retriever(
        vectordb_path=vectordb_path,
        collection_name=collection_name,
    )

    # init Tavily client
    tavily_client = TavilyClient(api_key=tavily_api_key)

    tools = create_online_tools(
        retriever=retriever,
        tavily_client=tavily_client,
        retrieval_top_k=top_k,
        web_search_max_results=web_search_max_results,
    )

    # load system prompt
    system_prompt = load_prompt("system", prompts_dir)

    # LLM and tools
    llm = get_llm("generate", llm_config)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: OnlineAgentState) -> dict:
        """Call the LLM with tools."""
        messages = state["messages"]

        # add system prompt if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # build graph
    graph = StateGraph(OnlineAgentState)

    # add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_online_agent(
    graph: StateGraph[OnlineAgentState], query: str
) -> OnlineAgentState:
    """
    Run the online agent with a query.

    :param graph: Compiled StateGraph
    :param query: User query
    :return: Final state
    """
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
    }

    final_state = graph.invoke(initial_state)
    return final_state


def get_final_answer(state: OnlineAgentState) -> str:
    """
    Extract the final answer from the agent state.

    :param state: Final agent state
    :return: The assistant's final answer
    """
    messages = state["messages"]
    # find the last AI message that isn't a tool call
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                return msg.content
    return "No answer generated."
