import os
from typing import List

import google.genai as genai
from src.wikivoyage_textfile_to_chromadb import create_chromadb_collection_from_csv
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, Embeddings


class GoogleGenAIEmbeddingFunction(embedding_functions.EmbeddingFunction[Documents]):
    def __init__(self, client: genai.Client, model_name: str = "models/text-embedding-004", task_type: str = "RETRIEVAL_DOCUMENT"):
        self.client = client
        self.model_name = model_name
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        # Google GenAI API has a limit of 100 documents per batch
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config={"task_type": self.task_type}
            )
            all_embeddings.extend([list(ce.values) for ce in response.embeddings])
        return all_embeddings


def build_prompt(query: str, context: List[str]) -> str:
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (str).
    """

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

    # combine the prompts to output a single prompt string
    system = f"{base_prompt['content']} {user_prompt['content']}"

    return system


def get_gemini_response(client: genai.Client, query: str, context: List[str]) -> str:
    """
    Queries the Gemini API to get a response to the question.

    Args:
    client (genai.Client): The Google GenAI client.
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    response = client.models.generate_content(model="gemini-2.0-flash", contents=build_prompt(query, context))

    return response.text


# def main():
#     print("Hello from rag-and-roll!")
#     destinations = extract_structured_data(MAIN_ENTRY_URL)
#     create_update_chromadb_collection(destinations)


def main(
    # collection_name: str = "documents_collection", persist_directory: str = "."
) -> None:
    # Check if the GOOGLE_API_KEY environment variable is set. Prompt the user to set it if not.
    if "GOOGLE_API_KEY" not in os.environ:
        gapikey = input("Please enter your Google API Key: ")
        os.environ["GOOGLE_API_KEY"] = gapikey

    google_api_key = os.environ["GOOGLE_API_KEY"]
    client = genai.Client(api_key=google_api_key, http_options={"api_version": "v1"})

    # create embedding function
    embedding_function = GoogleGenAIEmbeddingFunction(
        client=client,
        model_name="models/text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT",
    )

    # Fetch data for the collection. In this case, we are fetching travel destination information from Wikivoyage.
    # travel_result = extract_structured_data(
    #     url=MAIN_ENTRY_URL,
    #     search_term="travel destinations",
    #     input_selector="#searchInput",
    #     schema_model=TravelDestination,
    #     prompt="Extract travel destination information from this webpage. Focus on the name, location, description, best time to visit, attractions, difficulty level, and duration in days for each destination.",
    # )

    # Get the collection.
    # travel_result_dict = travel_result.model_dump()
    collection = create_chromadb_collection_from_csv(
        file_path=f"{os.path.dirname(os.path.abspath(__file__))}/wikivoyage_data/wikivoyage-listings-en.csv",
        embedding_function=embedding_function,
    )
    # We use a simple input loop.
    while True:
        # Get the user's query
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        # Update task type for query
        embedding_function.task_type = "RETRIEVAL_QUERY"

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        sources = "\n".join(
            [
                f"{result['filename']}: line {result['line_number']}"
                for result in results["metadatas"][0]  # type: ignore
            ]
        )

        # Get the response from Gemini
        response = get_gemini_response(client, query, results["documents"][0])  # type: ignore

        # Output, with sources
        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")


if __name__ == "__main__":
    main()
