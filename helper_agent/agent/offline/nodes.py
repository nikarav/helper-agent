from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from helper_agent.agent.offline.graph import OfflineAgentState
from helper_agent.utilities.logger import get_logger
from helper_agent.utilities.utils import load_prompt
from helper_agent.vectorstore.retriever import Retriever

logger = get_logger("helper_agent")


class EvaluationOutput(BaseModel):
    """Structured output for answer evaluation."""

    is_confident: bool
    reasoning: str


class ReformulationOutput(BaseModel):
    """Structured output for query reformulation."""

    reformulated_query: str
    reasoning: str


def create_retrieve_node(retriever: Retriever, top_k: int = 5):
    """
    Create a retrieve node function.

    :param retriever: Retriever instance
    :param top_k: Number of documents to retrieve
    :return: Node function
    """

    def retrieve(state: OfflineAgentState) -> dict:
        """Retrieve relevant documents for the current query."""
        query = state.get("reformulated_query") or state["query"]
        logger.debug(f"Retrieving {top_k} documents for: {query}")

        docs = retriever.retrieve_texts(query, top_k=top_k)
        return {"retrieved_docs": docs}

    return retrieve


def create_generate_node(
    llm: ChatGoogleGenerativeAI,
    prompts_dir: str,
):
    """
    Create a generate node function.

    :param llm: LLM instance for generation
    :param prompts_dir: Directory containing prompt templates
    :return: Node function
    """
    prompt_template = load_prompt("generate", prompts_dir)

    def generate(state: OfflineAgentState) -> dict:
        """Generate an answer using retrieved context."""
        query = state.get("reformulated_query") or state["query"]
        docs = state["retrieved_docs"]

        context = "\n\n---\n\n".join(docs) if docs else "No relevant context found."

        messages = [
            SystemMessage(content=prompt_template.format(context=context)),
            HumanMessage(content=query),
        ]

        logger.debug(f"Generating answer for: {query}")
        response = llm.invoke(messages)
        return {"answer": response.content}

    return generate


def create_evaluate_node(
    llm: ChatGoogleGenerativeAI,
    prompts_dir: str,
):
    """
    Create an evaluate node function with structured output.

    :param llm: LLM instance for evaluation
    :param prompts_dir: Directory containing prompt templates
    :return: Node function
    """
    prompt_template = load_prompt("evaluate", prompts_dir)
    structured_llm = llm.with_structured_output(
        schema=EvaluationOutput.model_json_schema(),
        method="json_schema",
    )

    def evaluate(state: OfflineAgentState) -> dict:
        """Evaluate if the answer is well-supported by context."""
        query = state.get("reformulated_query") or state["query"]
        docs = state["retrieved_docs"]
        answer = state["answer"]

        context = "\n\n---\n\n".join(docs) if docs else "No context."

        prompt = prompt_template.format(
            context=context,
            question=query,
            answer=answer,
        )

        messages = [HumanMessage(content=prompt)]

        logger.debug("Evaluating answer confidence...")
        result = structured_llm.invoke(messages)

        retry_count = state.get("retry_count", 0) + 1

        logger.debug(
            f"Evaluation: confident={result['is_confident']}, "
            f"reasoning={result['reasoning']}, retry_count={retry_count}"
        )

        return {"is_confident": result["is_confident"], "retry_count": retry_count}

    return evaluate


def create_reformulate_node(
    llm: ChatGoogleGenerativeAI,
    prompts_dir: str,
):
    """
    Create a reformulate node function with structured output.

    :param llm: LLM instance for reformulation
    :param prompts_dir: Directory containing prompt templates
    :return: Node function
    """
    prompt_template = load_prompt("reformulate", prompts_dir)
    structured_llm = llm.with_structured_output(
        schema=ReformulationOutput.model_json_schema(),
        method="json_schema",
    )

    def reformulate(state: OfflineAgentState) -> dict:
        """Reformulate the query to improve retrieval."""
        query = state.get("reformulated_query") or state["query"]
        answer = state["answer"]

        prompt = prompt_template.format(
            question=query,
            answer=answer,
        )

        messages = [HumanMessage(content=prompt)]

        logger.debug("Reformulating query...")
        result = structured_llm.invoke(messages)

        logger.debug(
            f"Reformulated: {result['reformulated_query']}, "
            f"reasoning={result['reasoning']}"
        )

        return {"reformulated_query": result["reformulated_query"]}

    return reformulate
