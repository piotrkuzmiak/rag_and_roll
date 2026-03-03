import os
from typing import List

from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.wikivoyage_textfile_to_chromadb import create_chromadb_collection_from_csv


def build_prompt(query: str, context: List[str]) -> str:
    base_prompt = {
        "content": "You are a helpful assistant. Answer the question using only the"
        " provided context. If there is not enough information in the context,"
        ' say "I am not sure", then try to make a guess.'
        " Format the answer in readable paragraphs.",
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have:"
        f"{(' ').join(context)}",
    }
    return f"{base_prompt['content']} {user_prompt['content']}"


def get_openai_response(client: OpenAI, query: str, context: List[str]) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=build_prompt(query, context),
    )
    return response.output_text


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        openai_api_key = input("Please enter your OpenAI API Key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        response = get_openai_response(client, query, results["documents"][0])  # type: ignore

        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")


if __name__ == "__main__":
    main()
