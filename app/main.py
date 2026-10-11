from fastapi import FastAPI

from app.api.v1.routes.ask import router as ask_router
from app.api.v1.routes.documents import router as documents_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.search import router as search_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1/documents")
app.include_router(search_router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
