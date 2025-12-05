from typing import TypedDict


class OfflineAgentState(TypedDict):
    """State passed between nodes in the agent graph."""

    query: str  # user query
    reformulated_query: str  # current query (may be reformulated)
    retrieved_docs: list[str]  # retrieved context chunks
    answer: str  # generated answer
    is_confident: bool  # evaluation result
    retry_count: int  # loop counter
