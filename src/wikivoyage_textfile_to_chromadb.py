"""
This module provides a function to create a ChromaDB collection from a CSV file.

The main function in this module is `create_chromadb_collection_from_csv`, which
reads data from a CSV file in chunks, and populates a ChromaDB collection with
the data.
"""
import pandas as pd
import chromadb


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

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    # Initialize ChromaDB client and collection
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=0):
        # Ensure document_column is of string type, filling NaNs, and filter out empty strings
        chunk[document_column] = chunk[document_column].fillna("").astype(str)
        mask = chunk[document_column].str.strip() != ""
        filtered_chunk = chunk[mask]

        if filtered_chunk.empty:
            continue

        ids = filtered_chunk.index.astype(str).tolist()
        documents = filtered_chunk[document_column].tolist()

        # Add filename and line number to metadata for main.py to use
        # Use .copy() to avoid SettingWithCopyWarning
        metadatas_chunk = filtered_chunk.copy()
        metadatas_chunk["filename"] = file_path.split("/")[-1]
        metadatas_chunk["line_number"] = (
            metadatas_chunk.index + 2
        )  # +1 for 0-indexing, +1 for header

        current_metadata_cols = metadata_columns + ["filename", "line_number"]
        metadatas = metadatas_chunk[current_metadata_cols].to_dict(orient="records")

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection
