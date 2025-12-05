from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OnlineAgentState(TypedDict):
    """State for the online ReAct agent with message history."""

    messages: Annotated[list[AnyMessage], add_messages]
    query: str  # Original user query
