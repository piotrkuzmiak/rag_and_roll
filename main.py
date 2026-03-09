import os
from pathlib import Path
from dataclasses import dataclass

import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic_ai import Agent, RunContext

from src.wikivoyage_textfile_to_chromadb import create_chromadb_collection_from_csv


@dataclass
class RAGDeps:
    collection: object


SYSTEM_PROMPT = (
    "You are a travel assistant for Polish mountain attractions. "
    "Use the `search_attractions` tool to retrieve relevant data before answering. "
    "Answer using only the retrieved context. "
    'If context is insufficient, reply with "I am not sure" and then provide a brief best-effort guess.'
)


agent = Agent("openai:gpt-4o", system_prompt=SYSTEM_PROMPT, deps_type=RAGDeps)


@agent.tool
def search_attractions(ctx: RunContext[RAGDeps], query: str, n_results: int = 5) -> str:
    """Search mountain attractions in ChromaDB and return formatted context with sources."""
    results = ctx.deps.collection.query(
        query_texts=[query],
        n_results=max(1, min(n_results, 10)),
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return "No results found."

    context_parts: list[str] = []
    for idx, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        context_parts.append(
            "\n".join(
                [
                    f"Result {idx}",
                    f"description: {document}",
                    f"address: {metadata.get('address', '')}",
                    f"directions: {metadata.get('directions', '')}",
                    f"hours: {metadata.get('hours', '')}",
                    f"checkIn: {metadata.get('checkIn', '')}",
                    f"checkOut: {metadata.get('checkOut', '')}",
                    f"latitude: {metadata.get('latitude', '')}",
                    f"longitude: {metadata.get('longitude', '')}",
                    f"accessibility: {metadata.get('accessibility', '')}",
                    f"source: {metadata.get('filename', 'unknown')}: line {metadata.get('line_number', '?')}",
                ]
            )
        )

    return "\n\n".join(context_parts)


def main(force_reindex: bool = False) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API Key: ").strip()

    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device=embedding_device,
        normalize_embeddings=False,
    )

    csv_path = Path(__file__).resolve().parent / "polish_mountains_hiking_trails_fake.csv"
    collection = create_chromadb_collection_from_csv(
        file_path=str(csv_path),
        collection_name="polish_mountains_hiking_trails",
        embedding_function=sentence_transformer_ef,
        force_reindex=force_reindex,
    )
    deps = RAGDeps(collection=collection)

    try:
        while True:
            query = input("Query: ").strip()
            if not query:
                print("Please enter a question. Ctrl+C to Quit.\n")
                continue

            print("\nThinking...\n")
            user_prompt = (
                "Use the `search_attractions` tool with the user question, then answer.\n"
                f"Question: {query}\n"
                "Answer in clear, readable paragraphs and include key source lines when relevant."
            )
            result = agent.run_sync(user_prompt, deps=deps)

            print(result.output)
            print()
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
