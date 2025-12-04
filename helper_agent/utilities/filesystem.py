import csv
import json
from pathlib import Path
from typing import Any

from helper_agent.data.models import Document


def save_failed_chunks(failures: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save failed chunks to a file.

    :param failures: List of failed chunks
    :param output_path: Path to save the failed chunks
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


def load_failed_chunks(path: Path) -> list[dict[str, Any]]:
    """
    Load failed chunks from a file.

    :param path: Path to the failed chunks file
    :return: List of dictionaries containing failed chunk information
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_documents(input_path: Path) -> list[Document]:
    """
    Load documents from JSON file.

    :param input_path: Path to input JSON
    :return: List of document dictionaries
    """
    with open(input_path) as f:
        docs = [Document.from_dict(doc) for doc in json.load(f)]
    return docs


def save_documents(
    docs: list[Document],
    output_dir: Path | str,
    filename: str,
    formats: list[str],
) -> None:
    """
    Save documents to the specified output formats.

    :param docs: List of documents to save
    :param output_dir: Output directory (Path or string)
    :param filename: Filename to save
    :param formats: List of formats to save
    :return: None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == "json":
            output_path = output_dir / f"{filename}.json"
            docs_data = [doc.to_dict() for doc in docs]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
        elif fmt == "csv":
            output_path = output_dir / f"{filename}_metadata.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "title",
                        "source",
                        "source_file",
                        "category",
                        "char_count",
                        "word_count",
                    ],
                )
                writer.writeheader()
                for doc in docs:
                    writer.writerow(
                        {
                            "title": doc.title,
                            "source": doc.source,
                            "source_file": doc.source_file,
                            "category": doc.category,
                            "char_count": doc.char_count,
                            "word_count": doc.word_count,
                        }
                    )
