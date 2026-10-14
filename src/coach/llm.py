"""Client for an OpenAI-compatible endpoint serving Apertus (vLLM, Ollama, llama.cpp).

Every call goes through `chat()`, which keeps the number of LLM calls per answer easy to count.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from .config import get_settings

log = logging.getLogger("coach.llm")


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout, max_retries=s.llm_max_retries)


def chat(
    messages: list[dict[str, str]],
    *,
    purpose: str,
    json_mode: bool = False,
    max_tokens: int = 400,
    temperature: float | None = None,
) -> str:
    """One LLM call. `purpose` is used for logging and by the mock backend."""
    s = get_settings()
    if s.mock_llm:
        return _mock(purpose, messages)
    kwargs: dict[str, Any] = {
        "model": s.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": s.llm_temperature if temperature is None else temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    log.debug("[%s] %s", purpose, text[:300])
    return text


def parse_json(text: str) -> dict[str, Any]:
    """Extract a JSON object even if the model wraps it in prose or code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    log.warning("Could not parse JSON from model output: %s", text[:200])
    return {}


def _mock(purpose: str, messages: list[dict[str, str]]) -> str:
    """Deterministic fake responses for testing the graph without a model."""
    last = messages[-1]["content"]
    if purpose == "assess":
        short = len(last.split("ANSWER:")[-1].split()) < 8
        return json.dumps(
            {
                "is_question_to_company": "?" in last.split("ANSWER:")[-1],
                "evidence_quote": last.split("ANSWER:")[-1].strip()[:80],
                "scores": {"clarity": 2 if short else 3, "relevance": 3},
                "strength": "mock strength",
                "improvement": "mock improvement",
                "coach_action": "probe" if short else "continue",
                "follow_up": "Kannst du mir dazu ein konkretes Beispiel geben?",
                "problem_flags": [],
            }
        )
    if purpose == "feedback":
        return "| Kriterium | Stufe | Beleg | Begründung | Tipp |\n|---|---|---|---|---|\n| Klarheit | Gut | «…» | … | … |\n\n(Mock-Feedback)"
    m = re.search(r"NEXT QUESTION: (.*)", last)
    return f"(mock) {m.group(1) if m else 'Okay.'}"
