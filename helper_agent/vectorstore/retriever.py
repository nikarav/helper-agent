from typing import Any

from helper_agent.vectorstore.embedder import GeminiEmbedder
from helper_agent.vectorstore.vectordb import ChromaVectorDB


class Retriever:
    """Retriever for querying the vector database."""

    def __init__(
        self,
        vectordb_path: str,
        collection_name: str = "langgraph_docs",
        embedding_model: str = "gemini-embedding-001",
        embedding_dimension: int = 512,
    ) -> None:
        """
        Initialize the retriever.

        :param vectordb_path: Path to ChromaDB persistent storage
        :param collection_name: Name of the collection
        :param embedding_model: Gemini embedding model name
        :param embedding_dimension: Embedding dimension
        """

        self._embedder = GeminiEmbedder(
            model=embedding_model,
            dimension=embedding_dimension,
            task_type="RETRIEVAL_QUERY",
        )

        self._vectordb = ChromaVectorDB(
            persist_directory=vectordb_path,
            collection_name=collection_name,
        )

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        :param query: Query string
        :param top_k: Number of documents to retrieve
        :return: List of retrieved documents with content and metadata
        """
        query_embedding = self._embedder.embed_query(query)

        results = self._vectordb.query(
            query_embedding=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None
                docs.append(
                    {
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

        return docs

    def retrieve_texts(self, query: str, top_k: int) -> list[str]:
        """
        Retrieve relevant document texts only.

        :param query: Query string
        :param top_k: Number of documents to retrieve
        :return: List of document content strings
        """
        return [doc["content"] for doc in self.retrieve(query, top_k)]
