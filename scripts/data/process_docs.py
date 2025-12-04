import argparse
import logging
from pathlib import Path

from helper_agent.data.filters import filter_by_categories
from helper_agent.data.parsers import parse_file
from helper_agent.utilities.configs import DotDict, load_configurations, print_config
from helper_agent.utilities.filesystem import save_documents
from helper_agent.utilities.logger import get_logger, set_log_level
from helper_agent.utilities.utils import print_summary

logger = get_logger("helper_agent")


def _extract_category_patterns(
    category_patterns: list[tuple[str, str]] | None,
) -> list[tuple[str, str]] | None:
    """
    Extract category patterns from source config.

    :param category_patterns: List of (pattern, category) tuples
    :return: List of (pattern, category) tuples or None
    """
    if not category_patterns:
        return None
    return [(pattern.pattern, pattern.category) for pattern in category_patterns]


def main(config: DotDict) -> None:
    """
    Process documentation files according to config.

    :param config: Configuration dictionary
    :return: None
    """
    logger.debug(f"Configuration:\n{print_config(config)}")

    all_docs = []
    for source_config in config.sources:
        filepath = Path(source_config.path)
        format_type = source_config.format
        source_name = source_config.name

        category_patterns = _extract_category_patterns(
            source_config.get("category_patterns")
        )

        docs = parse_file(filepath, format_type, source_name, category_patterns)
        logger.debug(f"  Parsed: {len(docs)} documents")

        filter_config = source_config.get("filter")
        if filter_config:
            include_cats = filter_config.get("include_categories")
            exclude_cats = filter_config.get("exclude_categories")
            docs = filter_by_categories(docs, include_cats, exclude_cats)
            logger.info(f"  Filtered: {len(docs)} documents from {source_name}")

        all_docs.extend(docs)

    print_summary(all_docs)
    save_documents(
        all_docs, config.output.directory, config.output.filename, config.output.formats
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process documentation files according to config."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/data_processing.yaml"),
        help="Path to config file (default: configs/data_processing.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        required=False,
        default=False,
        action="store_true",
        help="increase output verbosity",
    )
    args = parser.parse_args()
    if args.verbose:
        set_log_level(logger=logger, level=logging.DEBUG)
    main(load_configurations(args.config))
