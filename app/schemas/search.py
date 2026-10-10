from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    study_space_id: str = "dev-study-space"
    limit: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    document_id: str
    study_space_id: str
    page_number: int
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
