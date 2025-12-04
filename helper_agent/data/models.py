import hashlib
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class Document:
    """Parsed document from llms.txt files."""

    title: str
    source: str
    content: str
    source_file: str
    category: str

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "char_count": self.char_count,
            "word_count": self.word_count,
        }

    def __repr__(self) -> str:
        title = f"{self.title[:40]}..." if len(self.title) > 40 else self.title
        return f"Document(title='{title}', chars={self.char_count})"

    @staticmethod
    def from_dict(doc: dict) -> "Document":
        field_names = {f.name for f in fields(Document)}
        filtered = {key: value for key, value in doc.items() if key in field_names}
        return Document(**filtered)


@dataclass
class Chunk:
    """A document chunk with metadata."""

    content: str
    document: Document
    section_headers: dict[str, str]  # e.g., {"h2": "Setup", "h3": "Installation"}
    chunk_index: int = 0
    total_chunks: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for storage (flattened for ChromaDB)."""
        return {
            "content": self.content,
            "doc_title": self.document.title,
            "source": self.document.source,
            "source_file": self.document.source_file,
            "category": self.document.category,
            "section_headers": self.section_headers,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }

    def __repr__(self) -> str:
        title = self.document.title[:30]
        return f"Chunk({self.chunk_index}/{self.total_chunks}, doc='{title}...', {len(self.content)} chars)"

    def generate_id(self) -> str:
        """Generate unique ID for this chunk."""
        content_hash = hashlib.md5(
            f"{self.document.source}:{self.chunk_index}:{self.content[:100]}".encode()
        ).hexdigest()[:12]
        return f"{self.document.source_file}_{content_hash}_{self.chunk_index}"
