import argparse
from pathlib import Path

from helper_agent.data.filters import filter_by_categories
from helper_agent.data.parsers import parse_file
from helper_agent.utilities.configs import DotDict, load_configurations
from helper_agent.utilities.filesystem import save_documents
from helper_agent.utilities.logger import get_logger
from helper_agent.utilities.utils import print_summary

logger = get_logger(__name__)


def main(config: DotDict) -> None:
    """
    Process a single source file according to its config.

    :param source_config: Source configuration dictionary
    :return: None
    """
    all_docs = []
    for source_config in config.sources:
        filepath = Path(source_config.path)
        format_type = source_config.format
        source_name = source_config.name

        docs = parse_file(filepath, format_type, source_name)
        logger.info(f"  Parsed: {len(docs)} documents")

        filter_config = source_config.filter
        include_cats = filter_config.include_categories
        exclude_cats = filter_config.exclude_categories

        docs = filter_by_categories(docs, include_cats, exclude_cats)
        logger.info(f"  Filtered: {len(docs)} documents")

        all_docs.extend(docs)
    print_summary(all_docs)

    logger.info("\nSaving output...")
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
    main(load_configurations(args.config))
