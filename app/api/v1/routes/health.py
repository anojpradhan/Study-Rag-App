from fastapi import APIRouter

from app.services.memory_service import MemoryService
from app.vectorstore.qdrant_service import QdrantService

router = APIRouter(tags=["Health"])

qdrant_service = QdrantService()
memory_service = MemoryService()


@router.get("/health")
def health_check():
    dependencies = {}

    try:
        qdrant_service.client.get_collections()
        dependencies["qdrant"] = "healthy"
    except Exception:
        dependencies["qdrant"] = "unhealthy"

    try:
        memory_service.client.ping()
        dependencies["redis"] = "healthy"
    except Exception:
        dependencies["redis"] = "unhealthy"

    is_healthy = all(status == "healthy" for status in dependencies.values())

    return {
        "status": "healthy" if is_healthy else "degraded",
        "dependencies": dependencies,
    }
