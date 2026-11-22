import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.qdrant_service import QdrantService

STUDY_SPACE_ID = "fe0432cb-9fc9-43b4-a180-f9a2bd22a9f5"

embedding_service = EmbeddingService()
qdrant_service = QdrantService()

queries = [
    "Summarize the main topics covered in my uploaded document.",
    "What is machine learning?",
    "Explain supervised and unsupervised learning.",
    "What are IaaS, PaaS, and SaaS?",
    "Explain cloud computing and its service models.",
]

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = qdrant_service.search(
        query_embedding=embedding_service.embed_text(query),
        study_space_id=STUDY_SPACE_ID,
        limit=5,
    )

    if not results:
        print("No results returned from Qdrant.")
        continue

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", result["score"])
        print("Document ID:", result["document_id"])
        print("Text:", result["text"])

        if result["score"] < 0.35:
            print("WARNING: Below RAG relevance threshold (0.35)")
        else:
            print("PASS: Above RAG relevance threshold")


print("\n" + "=" * 80)
print("RETRIEVAL DIAGNOSTIC COMPLETED")
print("=" * 80)
