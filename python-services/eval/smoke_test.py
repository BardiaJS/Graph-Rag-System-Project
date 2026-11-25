import json
import math
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent / "backend"

load_dotenv(ROOT / ".env")
load_dotenv(BACKEND / ".env")

PYTHON_URL = os.getenv(
    "PYTHON_PROCESSOR_URL", "http://127.0.0.1:8001"
).rstrip("/")

LARAVEL_URL = os.getenv(
    "APP_URL", "http://127.0.0.1:8000"
).rstrip("/")

API_KEY = os.getenv("INTERNAL_API_KEY") or os.getenv("GRAPH_RAG_API_KEY")
DOCUMENT_ID = int(os.getenv("SMOKE_TEST_DOCUMENT_ID", "1"))
QUERY = os.getenv("SMOKE_TEST_QUERY", "PHP programming paradigms")

results = []


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append(passed)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def request_json(url, method="GET", payload=None, headers=None, timeout=20):
    data = None
    request_headers = dict(headers or {})

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    req = Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )

    with urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def main():
    print("GraphRAG Smoke Test")
    print("=" * 50)

    # 1. Configuration
    config_ok = bool(API_KEY and DOCUMENT_ID > 0)
    report(
        "Configuration",
        config_ok,
        "API key configured and document ID valid"
        if config_ok else "Missing API key or invalid document ID",
    )

    # 2. Laravel health endpoint
    try:
        req = Request(f"{LARAVEL_URL}/up", method="GET")
        with urlopen(req, timeout=10) as response:
            status = response.status
        report("Laravel /up", 200 <= status < 300, f"HTTP {status}")
    except Exception as exc:
        report("Laravel /up", False, str(exc)[:200])

    if not API_KEY:
        report("Python /health", False, "API key is not configured")
        report("Core dependencies", False, "Cannot inspect without API key")
        report("Python /retrieve", False, "Cannot authenticate")
        return

    headers = {"X-API-Key": API_KEY}

    # 3. Python health endpoint
    health = None
    try:
        status, health = request_json(
            f"{PYTHON_URL}/health",
            headers=headers,
            timeout=20,
        )
        report("Python /health", 200 <= status < 300, f"HTTP {status}")
    except Exception as exc:
        report("Python /health", False, str(exc)[:200])

    # 4. Core infrastructure
    if isinstance(health, dict):
        components = health.get("components", {})
        required = ("redis", "qdrant", "neo4j", "ollama")
        missing = [
            name for name in required
            if not isinstance(components.get(name), dict)
            or components[name].get("ok") is not True
        ]

        report(
            "Core dependencies",
            not missing,
            "All core components healthy"
            if not missing else "Unhealthy/missing: " + ", ".join(missing),
        )

        for warning in health.get("warnings", []):
            print(f"[WARN] {warning}")
    else:
        report("Core dependencies", False, "Health response unavailable")

    # 5. Functional retrieval + document scope
    try:
        status, body = request_json(
            f"{PYTHON_URL}/retrieve",
            method="POST",
            headers=headers,
            payload={
                "document_ids": [DOCUMENT_ID],
                "queries": [
                    {
                        "q": QUERY,
                        # Deliberately includes an unauthorized document ID.
                        "document_ids": [DOCUMENT_ID, 2147483647],
                    }
                ],
            },
            timeout=45,
        )

        evidence = body.get("evidence", [])
        valid = (
            200 <= status < 300
            and body.get("status") == "success"
            and isinstance(evidence, list)
            and len(evidence) > 0
        )

        report(
            "Python /retrieve",
            valid,
            f"HTTP {status}; {len(evidence)} evidence item(s)",
        )

        schema_ok = True
        scope_ok = True

        for item in evidence:
            required_fields = (
                "key", "document_id", "chunk_index",
                "text", "score", "headings",
                "linked_via", "figures",
            )

            if not isinstance(item, dict) or not all(
                field in item for field in required_fields
            ):
                schema_ok = False
                continue

            try:
                score_ok = math.isfinite(float(item["score"]))
            except (TypeError, ValueError):
                score_ok = False

            if (
                not isinstance(item["text"], str)
                or not item["text"].strip()
                or not isinstance(item["headings"], list)
                or not isinstance(item["linked_via"], list)
                or not isinstance(item["figures"], list)
                or not score_ok
            ):
                schema_ok = False

            if item["document_id"] != DOCUMENT_ID:
                scope_ok = False

        report("Evidence structure", valid and schema_ok)
        report(
            "Document scope",
            valid and scope_ok,
            f"Only document {DOCUMENT_ID} returned"
            if scope_ok else "Evidence contains an unauthorized document",
        )

    except Exception as exc:
        report("Python /retrieve", False, str(exc)[:200])
        report("Evidence structure", False, "Retrieval unavailable")
        report("Document scope", False, "Retrieval unavailable")

    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
