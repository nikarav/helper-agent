import argparse
import logging
import os
from typing import Any

from tqdm import tqdm

from helper_agent.data.chunkers import DocumentChunker
from helper_agent.data.cleaners import clean_document
from helper_agent.data.models import Chunk, Document
from helper_agent.utilities.configs import DotDict, load_configurations, print_config
from helper_agent.utilities.filesystem import (
    load_documents,
    load_failed_chunks,
    save_failed_chunks,
)
from helper_agent.utilities.logger import get_logger, set_log_level
from helper_agent.vectorstore.embedder import GeminiEmbedder
from helper_agent.vectorstore.vectordb import ChromaVectorDB

logger = get_logger("helper_agent")


def process_document(doc: Document, chunker: DocumentChunker) -> list[Chunk]:
    """
    Process a single document: clean and chunk.

    :param doc: Document dictionary
    :param chunker: DocumentChunker instance
    :return: List of chunks
    """
    content = doc.content
    source_file = doc.source_file

    if not content or not content.strip():
        logger.warning(f"Skipping empty document: {doc.title}")
        return []

    try:
        cleaned = clean_document(content, source_file)  # clean content
        if not cleaned or not cleaned.strip():
            logger.warning(f"Document empty after cleaning: {doc.title}")
            return []

        return chunker.chunk(doc)
    except Exception as e:
        logger.error(f"Error processing document '{doc.title}': {e}")
        return []


def _serialize_chunk(chunk: Chunk) -> dict[str, Any]:
    """
    Serialize a chunk to a dictionary.

    :param chunk: Chunk object
    :return: Dictionary containing chunk information
    """
    metadata = chunk.to_dict()
    metadata.pop("content", None)
    return {
        "id": chunk.generate_id(),
        "content": chunk.content,
        "metadata": metadata,
    }


def main(
    config: DotDict,
    resume_file: str | None = None,
    failed_chunks_filename: str = "failed_chunks.json",
) -> None:
    """
    Build vector database from documents.

    :param config: Configuration dictionary
    :param resume_file: Path to resume file
    :param failed_chunks_filename: Name of the failed chunks file
    """
    logger.debug(f"Configuration:\n{print_config(config)}")

    embed_cfg = config.embedding
    embedder = GeminiEmbedder(
        model=embed_cfg.model,
        dimension=embed_cfg.dimension,
        task_type=embed_cfg.task_type,
        rpm=config.rate_limits.rpm,
        tpm=config.rate_limits.tpm,
        batch_size=embed_cfg.batch_size,
    )
    logger.debug(
        f"Initialized GeminiEmbedder: model={embed_cfg.model}, dim={embed_cfg.dimension}, "
    )

    vectordb = ChromaVectorDB(
        persist_directory=config.output.vectordb_path,
        collection_name=config.output.collection_name,
    )
    logger.debug(
        f"Initialized ChromaDB: path={config.output.vectordb_path}, collection={config.output.collection_name}"
    )

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "skipped_docs": 0,
        "total_chunks": 0,
        "embedded_chunks": 0,
        "errors": 0,
    }

    payloads = []
    if resume_file:
        payloads = load_failed_chunks(resume_file)
        stats["total_chunks"] = len(payloads)
        logger.debug("Loaded %s failed chunks from %s", len(payloads), resume_file)
    else:
        input_path = config.input.path
        documents = load_documents(input_path)
        stats["total_docs"] = len(documents)
        logger.debug(f"Loaded {len(documents)} documents from {input_path}")

        chunker = DocumentChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            model_name=config.chunking.model_name,
        )

        logger.debug("Processing documents into chunks...")
        all_chunks: list[Chunk] = []
        for doc in tqdm(documents, desc="Chunking"):
            chunks = process_document(doc, chunker)
            if not chunks:
                stats["skipped_docs"] += 1
                continue
            all_chunks.extend(chunks)
            stats["processed_docs"] += 1
        stats["total_chunks"] = len(all_chunks)
        logger.debug(
            "Generated %s chunks from %s documents",
            len(all_chunks),
            stats["processed_docs"],
        )
        payloads = [_serialize_chunk(chunk) for chunk in all_chunks]

    if not payloads:
        logger.warning("Nothing to embed. Exiting.")
        return

    failure_path = os.path.join(config.output.vectordb_path, failed_chunks_filename)

    failed_payloads = []
    batch_size = embed_cfg.batch_size

    logger.debug("Embedding %s chunks...", len(payloads))
    for start in tqdm(range(0, len(payloads), batch_size), desc="Embedding"):
        batch = payloads[start : start + batch_size]
        ids = [item["id"] for item in batch]
        texts = [item["content"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        try:
            embeddings = embedder.embed_texts(texts)
            vectordb.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            stats["embedded_chunks"] += len(batch)
        except Exception:
            logger.exception("Error embedding batch starting at index %s", start)
            stats["errors"] += 1
            failed_payloads.extend(batch)

    stats["final_db_count"] = vectordb.count()

    if failed_payloads:
        save_failed_chunks(failed_payloads, failure_path)

    logger.info("Build complete. Final DB count: %s", stats["final_db_count"])
    logger.debug("final stats: %s", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build vector database from processed documents."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/embedding.yaml",
        help="Path to config file (default: configs/embedding.yaml)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database before building",
    )
    parser.add_argument(
        "--resume-file",
        type=str,
        default=None,
        help="Resume embedding from a failed-chunks JSON file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        set_log_level(logger=logger, level=logging.DEBUG)

    config = load_configurations(args.config)

    if args.reset:
        logger.warning("Resetting vector database...")
        vectordb = ChromaVectorDB(
            persist_directory=config.output.vectordb_path,
            collection_name=config.output.collection_name,
        )
        vectordb.reset()

    main(config=config, resume_file=args.resume_file)
