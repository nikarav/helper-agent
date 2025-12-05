from google import genai
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from helper_agent.data.models import Chunk, Document


class DocumentChunker:
    """Document chunker using LangChain text splitters with Gemini token counting."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str = "gemini-embedding-001",
    ) -> None:
        """
        Initialize the DocumentChunker.

        :param chunk_size: Size of each chunk
        :param chunk_overlap: Overlap between chunks
        :param model_name: Name of the model to use for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self._client = genai.Client()

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
        Count tokens using Gemini API.

        :param text: Text to count tokens
        :return: Total tokens
        """
        response = self._client.models.count_tokens(
            model=self.model_name,
            contents=text,
        )
        return response.total_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Chunk a document using two-stage splitting:

        Stage 1: Split by markdown headers (##, ###)
        Stage 2: Apply size limits with code protection

        :param doc: Document to chunk
        :return: List of Chunk objects
        """
        if not doc.content or not doc.content.strip():
            return []

        # Stage 1: Split by headers
        header_docs = self._header_splitter.split_text(doc.content)

        # Stage 2: Apply size limits
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
