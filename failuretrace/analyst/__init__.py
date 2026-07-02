"""Hypothesis generation: deterministic fallback first, optional Ollama enrichment.

The fallback path alone satisfies the whole pipeline; the local LLM is strictly
additive and can never smuggle in C2+ support or an unjustified hard constraint (the
deterministic classifier remains the authority on category and causal support level,
and the Pydantic model re-validates every result).
"""

from .fallback import build_fallback
from .ollama_client import OllamaClient, OllamaConfig, OllamaError, load_ollama_config
from .prompt import build_prompt
from .service import analyze

__all__ = [
    "build_fallback",
    "analyze",
    "build_prompt",
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "load_ollama_config",
]
