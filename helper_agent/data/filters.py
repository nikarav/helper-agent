from helper_agent.data.models import Document


def filter_by_categories(
    docs: list[Document],
    include_categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
) -> list[Document]:
    """
    Filter documents by category.

    :param docs: List of documents to filter
    :param include_categories: If provided, only keep documents in these categories
    :param exclude_categories: If provided, remove documents in these categories
    :return: Filtered list of documents
    """
    if include_categories is None and exclude_categories is None:
        return docs

    filtered = []
    for doc in docs:
        if include_categories is not None:
            if doc.category not in include_categories:
                continue

        if exclude_categories is not None:
            if doc.category in exclude_categories:
                continue

        filtered.append(doc)
    return filtered


def get_category_counts(docs: list[Document]) -> dict[str, int]:
    """
    Count documents per category.

    :param docs: List of documents
    :return: Dictionary mapping category to count, sorted by count descending
    """
    counts = {}
    for doc in docs:
        counts[doc.category] = counts.get(doc.category, 0) + 1

    return dict(
        sorted(counts.items(), key=lambda x: x[1], reverse=True)
    )  # sort by count descending


def get_source_counts(docs: list[Document]) -> dict[str, int]:
    """
    Count documents per source file.

    :param docs: List of documents
    :return: Dictionary mapping source_file to count
    """
    counts = {}
    for doc in docs:
        counts[doc.source_file] = counts.get(doc.source_file, 0) + 1
    return counts
