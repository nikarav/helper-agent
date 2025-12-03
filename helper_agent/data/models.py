from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Document:
    """Parsed document from llms.txt files."""

    title: str
    source: str  # file path or URL
    content: str
    source_file: (
        str  # which llms.txt file it came from (e.g., "langgraph", "langchain")
    )
    category: str  # extracted category for filtering

    @property
    def char_count(self) -> int:
        """Return character count of content."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Return word count of content."""
        return len(self.content.split())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "char_count": self.char_count,
            "word_count": self.word_count,
        }

    def __repr__(self) -> str:
        """Return a string representation of the document."""
        if len(self.title) > 40:
            title_preview = f"{self.title[:40]}..."
        else:
            title_preview = self.title
        return f"Document(title='{title_preview}', chars={self.char_count})"
