import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

collection_name = "study_chunks"

if not client.collection_exists(collection_name):
    print("Collection does not exist!")
    raise SystemExit(1)

info = client.get_collection(collection_name)

print("\n--- Collection Information ---")
print("Points:", info.points_count)
print("Vectors:", info.config.params.vectors)

print("\n--- Sample Points ---")

points, next_offset = client.scroll(
    collection_name=collection_name,
    limit=5,
    with_payload=True,
    with_vectors=False,
)

for point in points:
    print("\nPoint ID:", point.id)
    print("Payload:", point.payload)

print("\n--- End ---")
