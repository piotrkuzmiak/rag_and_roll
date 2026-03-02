import os
from typing import List

import google.genai as genai
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.wikivoyage_textfile_to_chromadb import create_chromadb_collection_from_csv


def build_prompt(query: str, context: List[str]) -> str:
    base_prompt = {
        "content": "I am going to ask you a question, which I would like you to answer"
        " based only on the provided context, and not any other information."
        " If there is not enough information in the context to answer the question,"
        ' say "I am not sure", then try to make a guess.'
        " Break your answer up into nicely readable paragraphs.",
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have:"
        f"{(' ').join(context)}",
    }
    return f"{base_prompt['content']} {user_prompt['content']}"


def get_gemini_response(client: genai.Client, query: str, context: List[str]) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=build_prompt(query, context)
    )
    return response.text


def main() -> None:
    if "GOOGLE_API_KEY" not in os.environ:
        gapikey = input("Please enter your Google API Key: ")
        os.environ["GOOGLE_API_KEY"] = gapikey

    google_api_key = os.environ["GOOGLE_API_KEY"]
    api_version = os.environ.get("GOOGLE_API_VERSION", "v1beta")
    client = genai.Client(api_key=google_api_key, http_options={"api_version": api_version})

    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=False
)

    collection = create_chromadb_collection_from_csv(
        file_path=f"{os.path.dirname(os.path.abspath(__file__))}/wikivoyage_data/wikivoyage-listings-en.csv",
        embedding_function=sentence_transformer_ef,
    )

    while True:
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        sources = "\n".join(
            [
                f"{result['filename']}: line {result['line_number']}"
                for result in results["metadatas"][0]  # type: ignore
            ]
        )

        response = get_gemini_response(client, query, results["documents"][0])  # type: ignore

        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")


if __name__ == "__main__":
    main()
