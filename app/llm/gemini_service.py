import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    answer: str
    model_used: str
    fallback_used: bool


class GeminiService:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.primary_model = settings.gemini_primary_model
        self.fallback_model = settings.gemini_fallback_model

    def _generate(self, model_name: str, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1500,
            ),
        )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError(f"Model '{model_name}' returned an empty response.")

        return answer.strip()

    def generate_answer(self, prompt: str) -> LLMResult:
        try:
            answer = self._generate(self.primary_model, prompt)
        except Exception:
            logger.exception(
                "Primary model '%s' failed. Trying fallback '%s'.",
                self.primary_model,
                self.fallback_model,
            )
        else:
            logger.info(
                "Generated answer using primary model: %s",
                self.primary_model,
            )
            return LLMResult(
                answer=answer,
                model_used=self.primary_model,
                fallback_used=False,
            )

        try:
            answer = self._generate(self.fallback_model, prompt)
        except Exception as exc:
            logger.exception("Both Gemini models failed.")
            raise RuntimeError(
                "Answer generation failed with both configured models."
            ) from exc

        logger.info(
            "Generated answer using fallback model: %s",
            self.fallback_model,
        )

        return LLMResult(
            answer=answer,
            model_used=self.fallback_model,
            fallback_used=True,
        )
