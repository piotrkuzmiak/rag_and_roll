"""
This module provides a function to create a ChromaDB collection from a CSV file.

The main function in this module is `create_chromadb_collection_from_csv`, which
reads data from a CSV file in chunks, and populates a ChromaDB collection with
the data.
"""
import pandas as pd
import chromadb
from pathlib import Path


def _count_csv_data_rows(file_path: str) -> int:
    """Count CSV data rows, excluding the header."""
    with open(file_path, "r", encoding="utf-8") as file:
        return max(sum(1 for _ in file) - 1, 0)


def _print_progress_bar(processed: int, total: int, embedded: int, width: int = 40) -> None:
    """Print a single-line terminal progress bar."""
    if total <= 0:
        return

    ratio = min(processed / total, 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rEmbedding progress [{bar}] {processed}/{total} rows ({ratio * 100:5.1f}%) | embedded: {embedded}",
        end="",
        flush=True,
    )


def create_chromadb_collection_from_csv(
    file_path: str,
    collection_name: str = "wikivoyage",
    document_column: str = "description",
    metadata_columns: list[str] = [
        "address",
        "directions",
        "hours",
        "checkIn",
        "checkOut",
        "latitude",
        "longitude",
        "accessibility",
    ],
    chunk_size: int = 500,
    embedding_function=None,
    show_progress: bool = True,
    persist_directory: str | Path = Path(__file__).resolve().parent.parent / "chroma_storage",
    force_reindex: bool = False,
) -> chromadb.Collection:
    """
    Creates or updates a ChromaDB collection from a CSV file.

    This function reads a CSV file in chunks, and for each chunk, it adds the
    data to a ChromaDB collection. The function allows specifying which columns
    to use for the document content and metadata.

    Args:
        file_path (str): The path to the CSV file.
        collection_name (str, optional): The name of the ChromaDB collection.
            Defaults to "wikivoyage".
        document_column (str, optional): The name of the column containing the
            document content. Defaults to "description".
        metadata_columns (list[str], optional): A list of column names to be
            used as metadata. Defaults to a predefined list of columns.
        chunk_size (int, optional): The number of rows to read from the CSV
            file at a time. Defaults to 500.
        persist_directory (str | Path, optional): Directory used by
            chromadb.PersistentClient. Defaults to
            <project_root>/chroma_storage.
        force_reindex (bool, optional): If True, re-read the CSV and upsert all
            rows even when the persistent collection already contains data.
            Defaults to False.

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    # Initialize persistent ChromaDB client and collection
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    existing_count = collection.count()

    # Reuse persistent embeddings when available, unless caller explicitly forces reindexing.
    if existing_count > 0 and not force_reindex:
        if show_progress:
            print(
                f"Using existing ChromaDB collection '{collection_name}' "
                f"with {existing_count} embedded documents."
            )
        return collection

    total_rows = _count_csv_data_rows(file_path) if show_progress else 0
    processed_rows = 0
    embedded_rows = 0

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=0):
        chunk_start_line_number = processed_rows + 2
        processed_rows += len(chunk)

        # Ensure document_column is of string type, filling NaNs, and filter out empty strings
        chunk[document_column] = chunk[document_column].fillna("").astype(str)
        mask = chunk[document_column].str.strip() != ""
        filtered_chunk = chunk[mask]
        filtered_line_numbers = [chunk_start_line_number + i for i, keep in enumerate(mask.tolist()) if keep]

        if show_progress:
            _print_progress_bar(processed_rows, total_rows, embedded_rows)

        if filtered_chunk.empty:
            continue
        ids = [str(line_number) for line_number in filtered_line_numbers]
        documents = filtered_chunk[document_column].tolist()

        # Add filename and line number to metadata for main.py to use
        # Use .copy() to avoid SettingWithCopyWarning
        metadatas_chunk = filtered_chunk.copy()
        metadatas_chunk["filename"] = Path(file_path).name
        metadatas_chunk["line_number"] = filtered_line_numbers

        current_metadata_cols = metadata_columns + ["filename", "line_number"]
        metadatas = metadatas_chunk[current_metadata_cols].to_dict(orient="records")

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        embedded_rows += len(filtered_chunk)

        if show_progress:
            _print_progress_bar(processed_rows, total_rows, embedded_rows)

    if show_progress and total_rows > 0:
        print()

    return collection
