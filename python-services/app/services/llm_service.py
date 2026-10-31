"""Minimal Ollama client. Local only: no document text leaves your infrastructure."""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class LLMError(RuntimeError):
    """The language model could not be reached or rejected the request (e.g. model not installed)."""


def _describe(e: Exception) -> str:
    """Include Ollama's own message: a bare '404 Not Found' hides 'model not found'."""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            body = resp.json().get("error") or resp.text
        except Exception:
            body = getattr(resp, "text", "")
        return f"HTTP {resp.status_code}: {str(body)[:200]}"
    return str(e)


def chat(
    messages: List[Dict[str, str]],
    json_mode: bool = False,
    temperature: float = 0.1,
    timeout: int = 180,
) -> str:
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def chat_json(
    messages: List[Dict[str, str]],
    default: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Ask for JSON.

    strict=False (default): never raises; returns `default` on any failure.
    strict=True: a transport/HTTP failure raises LLMError (so callers can tell
    "the model is down" from "the model found nothing"). Unparseable output is still
    returned as `default` in both modes."""
    try:
        raw = chat(messages, json_mode=True)
    except Exception as e:
        if strict:
            raise LLMError(_describe(e)) from e
        logger.error(f"LLM call failed: {_describe(e)}")
        return default if default is not None else {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning(f"Could not parse LLM JSON: {raw[:200]!r}")
        return default if default is not None else {}


def available_models() -> Optional[List[str]]:
    """Names of models installed in Ollama, or None if Ollama can't be reached."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        logger.warning(f"Could not list Ollama models at {OLLAMA_URL}: {_describe(e)}")
        return None


def check_model() -> Optional[str]:
    """A human-readable problem with the configured model, or None if it looks fine."""
    names = available_models()
    if names is None:
        return f"Ollama is not reachable at {OLLAMA_URL}; entity extraction will fail."
    wanted = {OLLAMA_MODEL, OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"}
    if not any(n in wanted for n in names):
        return (
            f"OLLAMA_MODEL '{OLLAMA_MODEL}' is not installed in Ollama. "
            f"Installed: {', '.join(names) or '(none)'}. Set OLLAMA_MODEL in the Python .env."
        )
    return None
