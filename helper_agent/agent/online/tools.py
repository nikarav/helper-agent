from langchain.tools import tool
from tavily import TavilyClient

from helper_agent.utilities.logger import get_logger
from helper_agent.vectorstore.retriever import Retriever

logger = get_logger("helper_agent")


def create_online_tools(
    *,
    retriever: Retriever,
    tavily_client: TavilyClient,
    retrieval_top_k: int = 5,
    web_search_max_results: int = 5,
) -> list:
    """
    Build tool instances bound to the provided dependencies.

    :param retriever: The retriever object
    :param tavily_client: The Tavily client object
    :param retrieval_top_k: The number of documents to retrieve from the retriever
    :param web_search_max_results: The number of results to retrieve from the web search
    :return: A list of tool instances
    """

    @tool(
        "search_documentation",
        description=(
            "Search the local LangChain / LangGraph knowledge base for API details "
            "or implementation guidance."
        ),
    )
    def search_documentation(query: str) -> str:
        logger.debug("Searching documentation for %s", query)
        docs = retriever.retrieve_texts(query, top_k=retrieval_top_k)
        if not docs:
            return "No relevant documentation found."
        return "\n\n---\n\n".join(docs)

    @tool(
        "web_search",
        description="Query Tavily for fresh, real-world information beyond local docs.",
    )
    def web_search(query: str) -> str:
        logger.debug("Web searching for %s", query)
        response = tavily_client.search(
            query=query,
            max_results=web_search_max_results,
            include_answer=True,
        )

        results = []
        summary = response.get("answer")
        if summary:
            results.append(f"Summary: {summary}")

        for result in response.get("results", []):
            title = result.get("title") or "Untitled result"
            content = result.get("content") or ""
            url = result.get("url") or ""
            body = f"**{title}**\n{content.strip()}"
            if url:
                body = f"{body}\nSource: {url}"
            results.append(body.strip())

        if not results:
            return "No web results found."

        return "\n\n---\n\n".join(results)

    return [search_documentation, web_search]
