import argparse
import logging
import os

from langgraph.graph import StateGraph

from helper_agent.agent.offline.graph import (
    OfflineAgentState,
    build_offline_graph,
    run_offline_agent,
)
from helper_agent.agent.online.graph import (
    OnlineAgentState,
    build_online_graph,
    get_final_answer,
    run_online_agent,
)
from helper_agent.utilities.configs import load_configurations
from helper_agent.utilities.logger import get_logger, set_log_level
from helper_agent.utilities.utils import print_answer_block, print_question_block

logger = get_logger("helper_agent")


def _run_offline_query(graph: StateGraph[OfflineAgentState], query: str) -> None:
    """
    Run a single offline query and print the result.

    :param graph: Compiled StateGraph
    :param query: User query
    """
    print_question_block(query)

    result = run_offline_agent(graph, query)

    print_answer_block(result["answer"])

    if result.get("retry_count", 0) > 1:
        logger.debug(f"\n(Reformulated {result['retry_count'] - 1} time(s))")

    if not result.get("is_confident", False) and result.get("retry_count", 0) >= 2:
        logger.debug(
            "\n(Note: Answer may not be fully supported by available documentation)"
        )


def _run_online_query(graph: StateGraph[OnlineAgentState], query: str) -> None:
    """
    Run a single online query and print the result.

    :param graph: Compiled StateGraph
    :param query: User query
    """
    print_question_block(query)

    result = run_online_agent(graph, query)
    answer = get_final_answer(result)

    print_answer_block(answer)


def _run_interactive(
    graph: StateGraph[OfflineAgentState | OnlineAgentState], mode: str
) -> None:
    """
    Run in interactive mode.

    :param graph: Compiled StateGraph
    :param mode: Operating mode
    """
    print(f"\nLangGraph Helper Agent - Interactive Mode ({mode})")
    print("Type 'quit' or 'exit' to stop.\n")

    run_query = _run_offline_query if mode == "offline" else _run_online_query

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            run_query(graph, query)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangGraph Helper Agent - Ask questions about LangGraph/LangChain"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Question to ask the agent",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["offline", "online"],
        default=None,
        help="Operating mode (default: from AGENT_MODE env var or 'offline')",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/agent.yaml",
        help="Path to agent config file (default: configs/agent.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Run in interactive mode (multiple queries)",
    )

    args = parser.parse_args()

    if args.verbose:
        set_log_level(logger=logger, level=logging.DEBUG)

    # explicit flag > env var > hardcoded default
    mode = args.mode or os.environ.get("AGENT_MODE", "offline")

    logger.info(f"Running in {mode} mode")

    config = load_configurations(args.config)
    mode_config = config[mode]

    if mode == "offline":
        retrieval_cfg = mode_config.retrieval
        agent_cfg = mode_config.get("agent", {})
        graph = build_offline_graph(
            prompts_dir=mode_config.get("prompts_dir"),
            llm_config=mode_config.llm,
            vectordb_path=retrieval_cfg.vectordb_path,
            collection_name=retrieval_cfg.collection_name,
            top_k=retrieval_cfg.get("top_k", 5),
            max_retries=agent_cfg.get("max_retries", 2),
        )
        run_query = _run_offline_query
    else:
        retrieval_cfg = mode_config.retrieval

        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable or web_search.api_key config required for online mode"
            )

        graph = build_online_graph(
            prompts_dir=mode_config.get("prompts_dir"),
            llm_config=mode_config.llm,
            vectordb_path=retrieval_cfg.vectordb_path,
            collection_name=retrieval_cfg.collection_name,
            tavily_api_key=tavily_api_key,
            top_k=retrieval_cfg.get("top_k", 5),
            web_search_max_results=mode_config.web_search.get("max_results", 5),
        )
        run_query = _run_online_query

    if args.interactive:
        _run_interactive(graph, mode)
    elif args.query:
        run_query(graph, args.query)
    else:
        parser.print_help()
        logger.error("\nError: Please provide a query or use --interactive mode.")
