from uuid import uuid4

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    study_space_id: str = "dev-study-space"
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=100,
    )
    limit: int = Field(default=5, ge=1, le=10)


class AnswerSource(BaseModel):
    document_id: str
    page_number: int
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[AnswerSource]
    model_used: str | None = None
    fallback_used: bool = False
    session_id: str
