import pytest

from app.services.entity_service import _parse


def test_extracts_only_entities_present_in_text():
    text = "Laravel uses Eloquent ORM and MySQL."

    raw = {
        "entities": [
            {"name": "Laravel", "type": "METHOD"},
            {"name": "Eloquent ORM", "type": "METHOD"},
            {"name": "MySQL", "type": "DATASET"},
            {"name": "PostgreSQL", "type": "DATASET"},
        ],
        "relations": [],
    }

    entities, relations = _parse(raw, text)

    names = {entity["name"] for entity in entities}

    assert names == {"Laravel", "Eloquent ORM", "MySQL"}
    assert relations == []


def test_rejects_invalid_entity_types():
    text = "Laravel uses Eloquent ORM."

    raw = {
        "entities": [
            {"name": "Laravel", "type": "METHOD"},
            {"name": "Eloquent ORM", "type": "UNKNOWN"},
        ],
        "relations": [],
    }

    entities, _ = _parse(raw, text)

    assert [entity["name"] for entity in entities] == ["Laravel"]


def test_deduplicates_entities():
    text = "Laravel is a PHP framework."

    raw = {
        "entities": [
            {"name": "Laravel", "type": "METHOD"},
            {"name": "Laravel", "type": "METHOD"},
        ],
        "relations": [],
    }

    entities, _ = _parse(raw, text)

    assert len(entities) == 1


def test_accepts_only_relations_between_extracted_entities():
    text = "Laravel uses Eloquent ORM and MySQL."

    raw = {
        "entities": [
            {"name": "Laravel", "type": "METHOD"},
            {"name": "Eloquent ORM", "type": "METHOD"},
            {"name": "MySQL", "type": "DATASET"},
        ],
        "relations": [
            {
                "source": "Laravel",
                "target": "Eloquent ORM",
                "type": "USES",
            },
            {
                "source": "Laravel",
                "target": "PostgreSQL",
                "type": "USES",
            },
            {
                "source": "Laravel",
                "target": "MySQL",
                "type": "INVALID",
            },
        ],
    }

    entities, relations = _parse(raw, text)

    assert len(entities) == 3
    assert len(relations) == 1
    assert relations[0]["type"] == "USES"


def test_empty_model_output_returns_empty_results():
    entities, relations = _parse({}, "Laravel uses Eloquent ORM.")

    assert entities == []
    assert relations == []


def test_entity_keys_are_normalized():
    text = "Laravel Framework"

    raw = {
        "entities": [
            {"name": "Laravel   Framework", "type": "METHOD"},
        ],
        "relations": [],
    }

    entities, _ = _parse(raw, text)

    assert entities[0]["key"] == "METHOD:laravel framework"
