import csv
import json
from pathlib import Path

from helper_agent.data.models import Document


def save_documents(
    docs: list[Document],
    output_dir: Path,
    filename: str,
    formats: list[str],
) -> None:
    """
    Save documents to the specified output formats.

    :param docs: List of documents to save
    :param output_dir: Output directory
    :param filename: Filename to save
    :param formats: List of formats to save
    :return: None
    """
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
