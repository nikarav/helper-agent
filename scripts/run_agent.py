import argparse
import logging
import os

from helper_agent.agent.offline.graph import build_offline_graph, run_offline_agent
from helper_agent.utilities.configs import load_configurations
from helper_agent.utilities.logger import get_logger, set_log_level

logger = get_logger("helper_agent")


def _run_single_query(graph, runner, query: str) -> None:
    """Run a single query and print the result."""
    print(f"\nQuestion: {query}\n")
    print("-" * 60)

    result = runner(graph, query)

    print(f"\nAnswer:\n{result['answer']}")

    if result.get("retry_count", 0) > 1:
        print(f"\n(Reformulated {result['retry_count'] - 1} time(s))")

    if not result.get("is_confident", False) and result.get("retry_count", 0) >= 2:
        print("\n(Note: Answer may not be fully supported by available documentation)")


def _run_interactive(graph, runner) -> None:
    """Run in interactive mode."""
    print("\nLangGraph Helper Agent - Interactive Mode")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            _run_single_query(graph, runner, query)
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
        default="offline",
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

    # arg mode > env var > default
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
        run_agent_fn = run_offline_agent
    else:
        # TODO: Implement online agent
        raise ValueError("Online mode not yet implemented")

    if args.interactive:
        _run_interactive(graph, run_agent_fn)
    elif args.query:
        _run_single_query(graph, run_agent_fn, args.query)
    else:
        parser.print_help()
        print("\nError: Please provide a query or use --interactive mode.")
