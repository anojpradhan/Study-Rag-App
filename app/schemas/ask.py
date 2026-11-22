from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    study_space_id: UUID
    conversation_id: UUID
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
    conversation_id: UUID
