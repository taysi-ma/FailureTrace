"""Thin client over a local Ollama-compatible endpoint.

Only used when ``ollama_enabled`` is true. ``requests`` is imported lazily so importing
FailureTrace never requires it. A ``session`` may be injected for testing. Any failure
(connection, timeout, HTTP error, malformed response) raises :class:`OllamaError`, which
the service turns into a deterministic fallback — the pipeline never crashes.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from ..core.settings import Settings

logger = logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout_seconds: float = 30.0
    max_retries: int = 1
    format: str = "json"


class OllamaError(RuntimeError):
    """Any failure talking to the Ollama endpoint."""


def load_ollama_config(settings: Settings) -> OllamaConfig:
    return OllamaConfig(**settings.section("ollama"))


class OllamaClient:
    """Minimal ``/api/generate`` client returning the model's text response."""

    def __init__(self, config: OllamaConfig, *, session=None) -> None:
        self.config = config
        self._session = session

    def _get_session(self):
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is an extra
            raise OllamaError(
                "requests is required for the Ollama client (pip install 'failuretrace[ollama]')"
            ) from exc
        self._session = requests.Session()
        return self._session

    def generate(self, prompt: str) -> str:
        """Return the model's raw text response (expected to be a JSON document)."""
        url = self.config.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": self.config.format,
        }
        session = self._get_session()
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = session.post(url, json=payload, timeout=self.config.timeout_seconds)
                status = getattr(response, "status_code", 200)
                if status != 200:
                    raise ValueError(f"Ollama returned HTTP {status}")
                data = response.json()
                text = data.get("response")
                if not isinstance(text, str):
                    raise ValueError("Ollama response missing a string 'response' field")
                return text
            except Exception as exc:  # noqa: BLE001 - all failures degrade to fallback
                last_exc = exc
                logger.warning(
                    "Ollama attempt %d/%d failed: %s",
                    attempt + 1, self.config.max_retries + 1, exc.__class__.__name__,
                )
        raise OllamaError(
            f"Ollama request failed after {self.config.max_retries + 1} attempt(s)"
        ) from last_exc
