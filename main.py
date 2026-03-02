import os
import time
from collections import deque
from typing import List

import google.genai as genai
from chromadb.api.types import Documents, Embeddings
from chromadb.utils import embedding_functions
from google.genai import errors as genai_errors

from src.wikivoyage_textfile_to_chromadb import create_chromadb_collection_from_csv


class GoogleGenAIEmbeddingFunction(embedding_functions.EmbeddingFunction[Documents]):
    def __init__(
        self,
        client: genai.Client,
        model_name: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        requests_per_minute: int = 100,
    ):
        self.client = client
        self.model_name = model_name
        self.task_type = task_type
        self.requests_per_minute = requests_per_minute
        self._request_timestamps: deque[float] = deque()

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        one_minute_ago = now - 60

        while self._request_timestamps and self._request_timestamps[0] <= one_minute_ago:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self.requests_per_minute:
            sleep_for = self._request_timestamps[0] + 60 - now
            if sleep_for > 0:
                time.sleep(sleep_for)

            now = time.monotonic()
            one_minute_ago = now - 60
            while self._request_timestamps and self._request_timestamps[0] <= one_minute_ago:
                self._request_timestamps.popleft()

    def _embed_batch_with_fallback(self, batch: Documents) -> list[list[float]]:
        models_to_try = [
            self.model_name,
            "gemini-embedding-001",
            "text-embedding-004",
        ]

        last_error = None
        for model in dict.fromkeys(models_to_try):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.embed_content(
                    model=model,
                    contents=batch,
                    config={"task_type": self.task_type},
                )
                self.model_name = model
                return [list(ce.values) for ce in response.embeddings]
            except genai_errors.ClientError as exc:
                # Retry only when the model is unavailable for the selected API version.
                if getattr(exc, "code", None) != 404:
                    raise
                last_error = exc
            finally:
                self._request_timestamps.append(time.monotonic())

        if last_error:
            raise last_error
        raise RuntimeError("Failed to generate embeddings for unknown reasons.")

    def __call__(self, input: Documents) -> Embeddings:
        # Google GenAI API has a limit of 100 documents per batch.
        batch_size = 100
        all_embeddings: Embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            all_embeddings.extend(self._embed_batch_with_fallback(batch))
        return all_embeddings


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

    embedding_function = GoogleGenAIEmbeddingFunction(
        client=client,
        model_name=os.environ.get("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"),
        task_type="RETRIEVAL_DOCUMENT",
        requests_per_minute=int(os.environ.get("GOOGLE_EMBEDDING_REQUESTS_PER_MINUTE", "100")),
    )

    collection = create_chromadb_collection_from_csv(
        file_path=f"{os.path.dirname(os.path.abspath(__file__))}/wikivoyage_data/wikivoyage-listings-en.csv",
        embedding_function=embedding_function,
    )

    while True:
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        embedding_function.task_type = "RETRIEVAL_QUERY"

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
