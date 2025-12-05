import tiktoken
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from helper_agent.data.models import Chunk, Document


class DocumentChunker:
    """Document chunker using LangChain text splitters with tiktoken token counting."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        greedy_merge: bool = True,
        model_name: str = "cl100k_base",
    ) -> None:
        """
        Initialize the DocumentChunker.

        Uses tiktoken for fast local token counting instead of API calls.
        cl100k_base encoding approximates Gemini tokenization with ~5% error,
        which is acceptable for chunking decisions.

        :param chunk_size: Size of each chunk in tokens
        :param chunk_overlap: Overlap between chunks in tokens
        :param greedy_merge: Whether to merge adjacent small sections within same h2 parent
        :param model_name: Tiktoken encoding name (default: cl100k_base)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.greedy_merge = greedy_merge
        self._encoding = tiktoken.get_encoding(model_name)

        # header splitter for markdown structure
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )

        # splitter with code protection
        self._size_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken (local, no API calls).

        :param text: Text to count tokens
        :return: Token count
        """
        return len(self._encoding.encode(text))

    def _merge_small_sections(self, header_docs: list) -> list:
        """
        Greedy merge adjacent sections that share the same h2 parent.

        Combines small sections to better utilize chunk_size while preserving
        topical coherence by only merging within the same parent header.

        :param header_docs: List of LangChain Document objects from header splitting
        :return: List of merged Document objects
        """
        if not header_docs:
            return []

        merged = []
        current = header_docs[0]

        for next_doc in header_docs[1:]:
            combined_content = current.page_content + "\n\n" + next_doc.page_content
            same_parent = current.metadata.get("h2") == next_doc.metadata.get("h2")

            if same_parent and self._count_tokens(combined_content) <= self.chunk_size:
                current.page_content = combined_content
            else:
                merged.append(current)
                current = next_doc

        merged.append(current)
        return merged

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Chunk a document using multi-stage splitting:

        Stage 1: Split by markdown headers (##, ###)
        Stage 2: (Optional) Merge small adjacent sections within same h2 parent
        Stage 3: Apply size limits with code protection

        :param doc: Document to chunk
        :return: List of Chunk objects
        """
        if not doc.content or not doc.content.strip():
            return []

        # Stage 1: Split by headers
        header_docs = self._header_splitter.split_text(doc.content)

        # Stage 2: Optionally merge small adjacent sections within same h2 parent
        if self.greedy_merge:
            header_docs = self._merge_small_sections(header_docs)

        # Stage 3: Apply size limits
        final_docs = self._size_splitter.split_documents(header_docs)

        total = len(final_docs)
        chunks = []
        for i, lc_doc in enumerate(final_docs):
            chunk = Chunk(
                content=lc_doc.page_content,
                document=doc,
                section_headers=lc_doc.metadata,
                chunk_index=i,
                total_chunks=total,
            )
            chunks.append(chunk)

        return chunks
