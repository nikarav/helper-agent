from helper_agent.data.filters import get_category_counts, get_source_counts
from helper_agent.data.models import Document
from helper_agent.utilities.logger import get_logger

logger = get_logger(__name__)


def print_summary(docs: list[Document]) -> None:
    """
    Print a summary of the processed documents.

    :param docs: List of documents to summarize
    :return: None
    """
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nTotal documents: {len(docs)}")
    total_chars = sum(doc.char_count for doc in docs)
    total_words = sum(doc.word_count for doc in docs)
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Total words: {total_words:,}")

    logger.info("\nBy source:")
    for source, count in get_source_counts(docs).items():
        source_docs = [d for d in docs if d.source_file == source]
        source_chars = sum(d.char_count for d in source_docs)
        logger.info(f"  {source}: {count} docs ({source_chars:,} chars)")

    logger.info("\nBy category:")
    for category, count in get_category_counts(docs).items():
        logger.info(f"  {category}: {count}")

    logger.info("=" * 60)
