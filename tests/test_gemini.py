from app.llm.gemini_service import GeminiService


def test_gemini_generation():
    service = GeminiService()

    result = service.generate_answer(
        "Explain retrieval-augmented generation in three sentences."
    )

    assert result.answer
    assert result.model_used
    assert isinstance(result.fallback_used, bool)

    print("Answer:", result.answer)
    print("Model used:", result.model_used)
    print("Fallback used:", result.fallback_used)


if __name__ == "__main__":
    test_gemini_generation()
