from unittest.mock import Mock

from app.rag.rag_service import RAGService


def make_chunk(score: float) -> dict:
    return {
        "text": "PaaS provides a platform for developing and deploying applications.",
        "document_id": "test-document",
        "study_space_id": "dev-study-space",
        "page_number": 1,
        "chunk_index": 0,
        "score": score,
    }


def test_irrelevant_chunks_are_filtered():
    retrieval = Mock()
    retrieval.retrieve.return_value = [make_chunk(score=0.10)]

    gemini = Mock()
    memory = Mock()
    memory.get_history.return_value = []

    service = RAGService(
        retrieval_service=retrieval,
        gemini_service=gemini,
        memory_service=memory,
    )

    result = service.answer_question(
        query="What are the best chess openings?",
        study_space_id="dev-study-space",
        session_id="test-session",
        limit=5,
    )

    assert result["sources"] == []
    assert "couldn't find relevant information" in result["answer"].lower()
    gemini.generate_answer.assert_not_called()
    memory.save_exchange.assert_called_once()


def test_relevant_chunks_generate_an_answer():
    retrieval = Mock()
    retrieval.retrieve.return_value = [make_chunk(score=0.68)]

    gemini = Mock()
    gemini.generate_answer.return_value = Mock(
        answer="PaaS provides a platform for developing and deploying applications.",
        model_used="test-model",
        fallback_used=False,
    )

    memory = Mock()
    memory.get_history.return_value = []

    service = RAGService(
        retrieval_service=retrieval,
        gemini_service=gemini,
        memory_service=memory,
    )

    result = service.answer_question(
        query="What does PaaS provide?",
        study_space_id="dev-study-space",
        session_id="test-session",
        limit=5,
    )

    assert result["answer"].startswith("PaaS provides")
    assert len(result["sources"]) == 1
    assert result["sources"][0]["score"] == 0.68
    gemini.generate_answer.assert_called_once()
    memory.save_exchange.assert_called_once()
