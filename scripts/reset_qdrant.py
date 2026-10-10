from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = "study_chunks"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)
    print(f"Deleted collection: {collection_name}")
else:
    print(f"Collection does not exist: {collection_name}")

print("The collection will be recreated when QdrantService initializes.")
