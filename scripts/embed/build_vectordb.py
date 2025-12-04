import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from helper_agent.data.chunkers import DocumentChunker
from helper_agent.data.cleaners import clean_document
from helper_agent.data.models import Chunk, Document
from helper_agent.embed.embedder import GeminiEmbedder
from helper_agent.embed.vectordb import ChromaVectorDB
from helper_agent.utilities.configs import DotDict, load_configurations, print_config
from helper_agent.utilities.filesystem import load_documents
from helper_agent.utilities.logger import get_logger, set_log_level

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


def main(config: DotDict) -> None:
    """
    Build vector database from documents.

    :param config: Configuration dictionary
    :return: None
    """
    logger.debug(f"Configuration:\n{print_config(config)}")

    input_path = Path(config.input.path)
    documents = load_documents(input_path)
    logger.debug(f"Loaded {len(documents)} documents from {input_path}")

    chunker = DocumentChunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        model_name=config.chunking.model_name,
    )

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
        "total_docs": len(documents),
        "processed_docs": 0,
        "skipped_docs": 0,
        "total_chunks": 0,
        "embedded_chunks": 0,
        "errors": 0,
    }

    logger.info("Processing documents into chunks...")
    all_chunks = []

    for doc in tqdm(documents, desc="Chunking"):
        chunks = process_document(doc, chunker)
        if not chunks:
            stats["skipped_docs"] += 1
            continue
        all_chunks.extend(chunks)
        stats["processed_docs"] += 1

    stats["total_chunks"] = len(all_chunks)
    logger.info(
        f"Generated {len(all_chunks)} chunks from {stats['processed_docs']} documents"
    )

    chunks_to_process = [(c, c.generate_id()) for c in all_chunks]

    logger.info(f"Embedding {len(chunks_to_process)} new chunks...")

    # Process in batches for embedding
    batch_size = embed_cfg.batch_size
    chunks_list = [c for c, _ in chunks_to_process]
    ids_list = [cid for _, cid in chunks_to_process]

    for i in tqdm(range(0, len(chunks_list), batch_size), desc="Embedding"):
        batch_chunks = chunks_list[i : i + batch_size]
        batch_ids = ids_list[i : i + batch_size]

        texts = [c.content for c in batch_chunks]

        try:
            embeddings = embedder.embed_texts(texts)
            metadatas = [c.to_dict() for c in batch_chunks]
            for m in metadatas:
                m.pop("content", None)

            vectordb.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            stats["embedded_chunks"] += len(batch_chunks)
        except Exception as e:
            logger.error(f"Error embedding batch at index {i}: {e}")
            stats["errors"] += 1

    stats["final_db_count"] = vectordb.count()
    logger.info(f"Build complete. Final DB count: {stats['final_db_count']}")

    logger.debug(f"final stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build vector database from processed documents."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/embedding.yaml"),
        help="Path to config file (default: configs/embedding.yaml)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database before building",
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
    main(config=config)
