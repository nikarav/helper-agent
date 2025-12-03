import re
from pathlib import Path

from helper_agent.data.models import Document


class LangGraphParser:
    """
    Parser for LangGraph llms-full.txt format.
    """

    @staticmethod
    def parse(filepath: Path, source_name: str = "langgraph") -> list[Document]:
        """
        Parse LangGraph llms-full.txt format.

        Format:
            ---
            path/to/doc.md
            ---
            # Title
            content...
            ---
            next/doc.md
            ---
            ...

        :param filepath: Path to the llms-full.txt file
        :param source_name: Name to identify this source (default: "langgraph")
        :return: List of parsed Document objects
        """
        content = filepath.read_text(encoding="utf-8")

        # Split by document delimiter pattern: ---\nfilepath.md\n---
        pattern = r"^---\n([\w\-/]+\.(?:md|ipynb))\n---\n"

        docs = []
        parts = re.split(pattern, content, flags=re.MULTILINE)

        # parts[0] is empty or preamble, then alternating: path, content, path, content...
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break

            path = parts[i].strip()
            doc_content = parts[i + 1].strip()

            # title from first # heading
            title_match = re.search(r"^#\s+(.+)$", doc_content, re.MULTILINE)
            title = title_match.group(1) if title_match else path

            # category from path (e.g., "how-tos", "concepts")
            category = path.split("/")[0] if "/" in path else "root"

            docs.append(
                Document(
                    title=title,
                    source=path,
                    content=doc_content,
                    source_file=source_name,
                    category=category,
                )
            )
        return docs


class LangChainParser:
    """
    Parser for LangChain llms-full.txt format.
    """

    @staticmethod
    def _extract_category(url: str) -> str:
        """
        Extract category from langchain URL for filtering.

        :param url: URL to extract category from
        :return: Category
        """
        category_patterns = [
            ("/oss/python/langchain/", "oss/python/langchain"),
            ("/oss/python/langgraph/", "oss/python/langgraph"),
            ("/oss/python/integrations/", "oss/python/integrations"),
            ("/oss/python/migrate/", "oss/python/migrate"),
            ("/oss/python/releases/", "oss/python/releases"),
            ("/oss/python/deepagents/", "oss/python/deepagents"),
            ("/oss/python/contributing/", "oss/python/contributing"),
            ("/oss/python/concepts/", "oss/python/concepts"),
            ("/oss/python/", "oss/python/other"),
            ("/oss/javascript/", "oss/javascript"),
            ("/langsmith/", "langsmith"),
        ]

        for pattern, category in category_patterns:
            if pattern in url:
                return category
        return "other"

    @staticmethod
    def parse(filepath: Path, source_name: str = "langchain") -> list[Document]:
        """
        Parse LangChain llms-full.txt format.

        Format:
            # Title
            Source: https://...

            content...

            ***

            # Next Title
            Source: https://...
            ...

        :param filepath: Path to the llms-full.txt file
        :param source_name: Name to identify this source (default: "langchain")
        :return: List of parsed Document objects
        """
        content = filepath.read_text(encoding="utf-8")

        # (# at start of line, followed by Source:)
        pattern = r"^(# .+?)\nSource: (https?://[^\n]+)\n"

        docs = []
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        for i, match in enumerate(matches):
            title = match.group(1).lstrip("# ").strip()
            source = match.group(2).strip()

            # Content is from end of this match to start of next (or end of file)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            doc_content = content[start:end].strip()

            doc_content = re.sub(r"\n\*{3}\s*$", "", doc_content)  # trailing separators
            category = LangChainParser._extract_category(source)
            docs.append(
                Document(
                    title=title,
                    source=source,
                    content=doc_content,
                    source_file=source_name,
                    category=category,
                )
            )
        return docs


def parse_file(
    filepath: Path | str,
    format_type: str,
    source_name: str | None = None,
) -> list[Document]:
    """
    Parse a documentation file using the appropriate parser.

    :param filepath: Path to the file to parse
    :param format_type: Type of format ("langgraph" or "langchain")
    :param source_name: Optional name for the source (defaults to format_type)
    :return: List of parsed Document objects
    :raises ValueError: If format_type is not supported
    """
    filepath = Path(filepath)
    source_name = source_name or format_type

    if format_type == "langgraph":
        return LangGraphParser.parse(filepath, source_name)
    elif format_type == "langchain":
        return LangChainParser.parse(filepath, source_name)
    else:
        raise ValueError(f"Unknown format type: {format_type}")
