import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("helper_agent")


class ChromaVectorDB:
    """ChromaDB wrapper for document embeddings storage."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "langgraph_docs",
    ) -> None:
        """
        Initialize ChromaDB with persistent storage.

        :param persist_directory: Directory for persistent storage
        :param collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Upsert documents (add or update if exists).

        :param ids: Unique identifiers for documents
        :param embeddings: Embedding vectors
        :param documents: Document texts
        :param metadatas: Optional metadata for each document
        """
        if not ids:
            return

        # Sanitize metadata
        if metadatas:
            metadatas = [self._sanitize_metadata(m) for m in metadatas]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Upserted {len(ids)} documents to collection")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query the collection by embedding.

        :param query_embedding: Query embedding vector
        :param n_results: Number of results to return
        :param where: Optional filter conditions
        :param include: What to include in results (documents, embeddings, metadatas, distances)
        :return: Query results
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include,
        )

        return results

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """
        Get which of the given IDs already exist in the collection.

        :param ids: List of IDs to check
        :return: Set of existing IDs
        """
        if not ids:
            return set()
        result = self._collection.get(ids=ids, include=[])
        return set(result["ids"])

    def reset(self) -> None:
        """Reset the collection (delete all documents)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Reset collection: {self.collection_name}")

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize metadata values for ChromaDB compatibility.

        ChromaDB only accepts str, int, float, bool as metadata values.

        :param metadata: Original metadata
        :return: Sanitized metadata
        """
        sanitized = {}

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = json.dumps(value)
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized
