import pytest

from app.services.verify_service import verify_answer


@pytest.fixture
def sources():
    return [
        {
            "sid": "S1",
            "text": (
                "PHP has eight data types. "
                "Four are scalar types and four are compound types."
            ),
            "figure_numbers": [],
        },
        {
            "sid": "S2",
            "text": (
                "PHP supports integers, floats, strings, and booleans "
                "as scalar data types."
            ),
            "figure_numbers": [],
        },
    ]


def test_valid_citation_is_preserved(sources):
    answer = "PHP has eight data types [S1]."

    result = verify_answer(answer, sources)

    assert result["answer"] == answer
    assert result["cited_sids"] == ["S1"]
    assert result["refusal"] is False


def test_invalid_citation_is_removed(sources):
    result = verify_answer(
        "PHP has eight data types [S9].",
        sources,
    )

    assert "[S9]" not in result["answer"]
    assert result["cited_sids"] == []
    assert any("non-existent sources" in warning for warning in result["warnings"])


def test_refusal_does_not_keep_citations(sources):
    result = verify_answer(
        "The provided sources do not contain this information [S1].",
        sources,
    )

    assert result["refusal"] is True
    assert result["cited_sids"] == []
    assert "[S1]" not in result["answer"]


def test_unsupported_figure_is_reported(sources):
    result = verify_answer(
        "See Figure 99 [S1].",
        sources,
    )

    assert "99" in result["unverified_figures"]


def test_existing_citation_does_not_trigger_repair_by_default(sources):
    result = verify_answer(
        "PHP has eight data types [S1].",
        sources,
    )

    assert result["answer"] == "PHP has eight data types [S1]."
    assert result["cited_sids"] == ["S1"]


def test_repair_is_disabled_by_default(sources, monkeypatch):
    from app.services import verify_service

    monkeypatch.setattr(
        verify_service,
        "CITATION_REPAIR_ENABLED",
        False,
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("chat_json should not be called when repair is disabled")

    monkeypatch.setattr(verify_service, "chat_json", unexpected_call)

    answer = "PHP has eight data types."
    result = verify_answer(answer, sources)

    assert result["answer"] == answer
    assert result["citation_repair"]["status"] == "disabled"
    assert result["citation_repair"]["attempted"] == 0


def test_repairs_uncited_english_sentence(sources, monkeypatch):
    from app.services import verify_service

    monkeypatch.setattr(
        verify_service,
        "CITATION_REPAIR_ENABLED",
        True,
    )
    monkeypatch.setattr(
        verify_service,
        "chat_json",
        lambda *args, **kwargs: {
            "repairs": [{"index": 0, "sids": ["S1"]}]
        },
    )

    result = verify_answer(
        "PHP has eight data types.",
        sources,
    )

    assert result["answer"] == "PHP has eight data types [S1]."
    assert result["cited_sids"] == ["S1"]
    assert result["citation_repair"]["repaired"] == 1


def test_repairs_uncited_persian_sentence(monkeypatch):
    from app.services import verify_service

    monkeypatch.setattr(
        verify_service,
        "CITATION_REPAIR_ENABLED",
        True,
    )
    monkeypatch.setattr(
        verify_service,
        "chat_json",
        lambda *args, **kwargs: {
            "repairs": [{"index": 0, "sids": ["S1"]}]
        },
    )

    sources = [
        {
            "sid": "S1",
            "text": "پی‌اچ‌پی هشت نوع داده دارد.",
            "figure_numbers": [],
        }
    ]

    result = verify_answer(
        "پی‌اچ‌پی هشت نوع داده دارد.",
        sources,
    )

    assert result["answer"] == "پی‌اچ‌پی هشت نوع داده دارد [S1]."
    assert result["cited_sids"] == ["S1"]
    assert result["citation_repair"]["repaired"] == 1


def test_invalid_repair_sid_is_rejected(sources, monkeypatch):
    from app.services import verify_service

    monkeypatch.setattr(
        verify_service,
        "CITATION_REPAIR_ENABLED",
        True,
    )
    monkeypatch.setattr(
        verify_service,
        "chat_json",
        lambda *args, **kwargs: {
            "repairs": [{"index": 0, "sids": ["S99"]}]
        },
    )

    result = verify_answer(
        "PHP has eight data types.",
        sources,
    )

    assert "[S99]" not in result["answer"]
    assert result["cited_sids"] == []
    assert result["citation_repair"]["repaired"] == 0
    assert result["citation_repair"]["unresolved"] == 1


def test_empty_repair_does_not_add_citation(sources, monkeypatch):
    from app.services import verify_service

    monkeypatch.setattr(
        verify_service,
        "CITATION_REPAIR_ENABLED",
        True,
    )
    monkeypatch.setattr(
        verify_service,
        "chat_json",
        lambda *args, **kwargs: {
            "repairs": [{"index": 0, "sids": []}]
        },
    )

    answer = "This statement is not supported by the sources."
    result = verify_answer(answer, sources)

    assert result["answer"] == answer
    assert result["cited_sids"] == []
    assert result["citation_repair"]["repaired"] == 0
    assert result["citation_repair"]["unresolved"] == 1


def test_existing_citation_is_preserved_when_repair_enabled(
    sources,
    monkeypatch,
):
    from app.services import verify_service

    monkeypatch.setattr(
        verify_service,
        "CITATION_REPAIR_ENABLED",
        True,
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("No repair should be attempted for an already cited sentence")

    monkeypatch.setattr(verify_service, "chat_json", unexpected_call)

    answer = "PHP has eight data types [S1]."
    result = verify_answer(answer, sources)

    assert result["answer"] == answer
    assert result["cited_sids"] == ["S1"]
    assert result["citation_repair"]["attempted"] == 0
