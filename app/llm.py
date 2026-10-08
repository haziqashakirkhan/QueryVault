"""Thin wrapper around the Groq chat API so the rest of the code never touches
the SDK directly (easy to swap for another provider)."""
from groq import Groq


class LLMError(Exception):
    """Any problem talking to the model, with a message safe to show the user."""


class LLMNotConfigured(LLMError):
    pass


class LLMClient:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> Groq:
        if not self.ready:
            raise LLMNotConfigured(
                "GROQ_API_KEY is missing. Copy .env.example to .env, add your key and restart."
            )
        if self._client is None:
            # max_retries: the SDK backs off and retries on rate limits (HTTP 429)
            self._client = Groq(api_key=self.api_key, max_retries=4)
        return self._client

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, json_mode=False) -> str:
        kwargs = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._get_client().chat.completions.create(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"The language model request failed: {exc}") from exc
        return (response.choices[0].message.content or "").strip()
