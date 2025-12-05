import os

from helper_agent.data.filters import get_category_counts, get_source_counts
from helper_agent.data.models import Document
from helper_agent.utilities.logger import get_logger

logger = get_logger("helper_agent")


def load_prompt(name: str, prompts_dir: str) -> str:
    """
    Load a prompt template from file.

    :param name: Prompt name (without .txt extension)
    :param prompts_dir: Directory containing prompts
    :return: Prompt template string
    """
    prompt_file = os.path.join(prompts_dir, f"{name}.txt")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    content = open(prompt_file, "r", encoding="utf-8").read()
    logger.debug(f"Loaded prompt: {name}")
    return content


def print_summary(docs: list[Document]) -> None:
    """
    Print a summary of the processed documents.

    :param docs: List of documents to summarize
    :return: None
    """
    logger.debug("\n" + "=" * 60)
    logger.debug("PROCESSING SUMMARY")
    logger.debug("=" * 60)

    logger.debug(f"\nTotal documents: {len(docs)}")
    total_chars = sum(doc.char_count for doc in docs)
    total_words = sum(doc.word_count for doc in docs)
    logger.debug(f"Total characters: {total_chars:,}")
    logger.debug(f"Total words: {total_words:,}")

    logger.debug("\nBy source:")
    for source, count in get_source_counts(docs).items():
        source_docs = [d for d in docs if d.source_file == source]
        source_chars = sum(d.char_count for d in source_docs)
        logger.debug(f"  {source}: {count} docs ({source_chars:,} chars)")

    logger.debug("\nBy category:")
    for category, count in get_category_counts(docs).items():
        logger.debug(f"  {category}: {count}")

    logger.debug("=" * 60)
