import pytest

from app.services import llm_service


def test_chat_json_returns_default_on_invalid_json_by_default(monkeypatch):
    monkeypatch.setattr(llm_service, "chat", lambda *args, **kwargs: "not json")

    assert llm_service.chat_json([], default={}) == {}


def test_chat_json_can_raise_on_invalid_json(monkeypatch):
    monkeypatch.setattr(llm_service, "chat", lambda *args, **kwargs: "not json")

    with pytest.raises(llm_service.LLMError):
        llm_service.chat_json([], strict=True, raise_on_invalid_json=True)
