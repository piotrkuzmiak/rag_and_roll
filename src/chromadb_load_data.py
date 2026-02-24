import json
import chromadb

def create_update_chromadb_collection(collection_name: str = "travel_destinations", travel_result: str = "") -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    data =json.loads(travel_result)
    collection.add(
        ids=[data.pop("name")],
        documents=[data.pop("description")],
        metadatas=[data]
    )
    return collection