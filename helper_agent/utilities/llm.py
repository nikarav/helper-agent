from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(purpose: str, llm_config: dict[str, Any]) -> ChatGoogleGenerativeAI:
    """
    Get LLM instance for a specific purpose.

    Merges default config with purpose-specific overrides.

    :param purpose: Node purpose ("generate", "evaluate", "reformulate")
    :param llm_config: LLM config dict with "default" and optional per-purpose overrides
    :return: Configured ChatGoogleGenerativeAI instance
    """
    default = llm_config.get("default", {})
    override = llm_config.get(purpose, {})
    settings = {**default, **override}

    return ChatGoogleGenerativeAI(
        model=settings.get("model", "gemini-2.0-flash"),
        temperature=settings.get("temperature", 0.7),
        max_output_tokens=settings.get("max_tokens", 1024),
    )
