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
) -> Dict[str, Any]:
    """Ask for JSON; never raise on malformed output, return `default` instead."""
    try:
        raw = chat(messages, json_mode=True)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
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
