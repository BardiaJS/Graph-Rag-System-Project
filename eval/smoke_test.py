#!/usr/bin/env python3
"""
Smoke test: one command that checks the whole stack and says what is wrong. Standard library only.
No LLM calls, nothing is modified or deleted, so it runs in seconds and is safe to repeat.

    export INTERNAL_API_KEY=...     # Python service secret (read it from the .env, do not type it)
    export LARAVEL_TOKEN=...        # Sanctum token of the test user:   read -rs LARAVEL_TOKEN
    python eval/smoke_test.py --doc-id 5 --user-id 1

Optional:
    export LARAVEL_TOKEN_OTHER=...  # token of a DIFFERENT user: proves other people's figures are refused
    --laravel-url / --rag-url       defaults http://localhost:8000 / http://localhost:8001

Each check prints PASS, FAIL, WARN (works but needs attention) or SKIP (not enough information given).
Exit code 1 if anything FAILED.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

RESULTS = []


def record(status, name, detail=""):
    RESULTS.append((status, name, detail))
    print(f"  {status:5s} {name}" + (f"  -> {detail}" if detail else ""))


def http(method, url, payload=None, headers=None, timeout=30, binary=False):
    """Returns (status, body) where body is parsed JSON, raw bytes (binary=True) or {} / text."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    except Exception as e:
        return 0, {"error": str(e)}
    if binary:
        return code, raw
    try:
        return code, json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        return code, {"raw": raw[:200].decode("utf-8", "replace")}


# ------------------------------------------------------------------------ Python service

def check_python(args, rag_headers):
    print("\nPython service")
    status, body = http("GET", f"{args.rag_url}/health", headers=rag_headers)
    if status == 0:
        record("FAIL", "service reachable", f"{body.get('error')} (is uvicorn running?)")
        return False
    if status == 401:
        record("FAIL", "API key accepted", "401: INTERNAL_API_KEY missing or different from the service's")
        return False

    comps = body.get("components", {})
    for name in ("redis", "qdrant", "neo4j"):
        c = comps.get(name, {})
        record("PASS" if c.get("ok") else "FAIL", f"{name} up", c.get("error", ""))
    record("PASS" if comps.get("ollama", {}).get("ok") else "WARN", "ollama model installed",
           "; ".join(w for w in body.get("warnings", []) if "OLLAMA" in w or "Ollama" in w))
    workers = comps.get("redis", {}).get("workers", {})
    record("PASS" if workers.get("ingestion") == 1 else "FAIL", "exactly one ingestion worker",
           f"found {workers.get('ingestion')}")
    record("PASS" if workers.get("entities", 0) >= 1 else "WARN", "entity worker running",
           "optional: python -m app.worker --entities")
    failed = sum(q.get("failed", 0) for q in comps.get("redis", {}).get("queues", {}).values())
    record("PASS" if not failed else "WARN", "no failed jobs in the queues", f"{failed} failed")

    s401, _ = http("GET", f"{args.rag_url}/health")  # no key at all
    record("PASS" if s401 == 401 else "WARN", "service rejects requests without the key",
           "" if s401 == 401 else f"got HTTP {s401}: INTERNAL_API_KEY is not enforced")
    return True


def check_verifier(args, rag_headers):
    print("\nAnswer verifier (the defence against invented answers and stuffed citations)")
    plain = [
        {"sid": "S1", "text": "Variables start with a dollar sign."},
        {"sid": "S2", "text": "The echo statement prints strings."},
        {"sid": "S3", "text": "Arrays store several values."},
    ]

    code, r = http("POST", f"{args.rag_url}/verify", {
        "answer": "The default port number for a MySQL server is 3306 [S1][S2][S3].", "sources": plain,
    }, rag_headers)
    ok = code == 200 and r.get("unsupported_numbers") == ["3306"]
    record("PASS" if ok else "FAIL", "flags a number found in no source",
           "" if ok else "no 'unsupported_numbers' in the reply: an OLD verifier is running; restart uvicorn")

    code, r = http("POST", f"{args.rag_url}/verify", {
        "answer": "The sources provided do not contain information about the capital of France. [S1][S2][S3]",
        "sources": plain,
    }, rag_headers)
    ok = code == 200 and r.get("refusal") is True and r.get("cited_sids") == [] and "[S" not in r.get("answer", "[S")
    record("PASS" if ok else "FAIL", "a refusal cites nothing", "" if ok else str(r)[:140])

    twelve = [{"sid": f"S{i}", "text": f"Unrelated passage number {i} about echo statements and comments."}
              for i in range(1, 13)]
    twelve[3]["text"] = "The foreach loop is used to iterate over elements in an array or other iterable objects."
    code, r = http("POST", f"{args.rag_url}/verify", {
        "answer": "The foreach loop in PHP is used to iterate over elements in an array " +
                  "".join(f"[S{i}]" for i in range(1, 13)) + ".",
        "sources": twelve,
    }, rag_headers)
    ok = code == 200 and r.get("cited_sids") == ["S4"]
    record("PASS" if ok else "FAIL", "prunes citation stuffing (12 cited -> the 1 that matches)",
           "" if ok else f"cited {r.get('cited_sids')}")


def check_document(args, rag_headers):
    print(f"\nDocument {args.doc_id} in the stores")
    if args.doc_id is None:
        record("SKIP", "document consistency", "give --doc-id")
        return
    code, body = http("GET", f"{args.rag_url}/documents/{args.doc_id}?user_id={args.user_id}", headers=rag_headers)
    if code == 404:
        record("FAIL", "document exists", "404: wrong --doc-id/--user-id, or it was deleted")
        return
    d = body.get("document", {})
    record("PASS" if d.get("consistent") else "FAIL", "Qdrant vectors match graph chunks",
           f"chunks={d.get('chunks')} qdrant_points={d.get('qdrant_points')}")
    ent = d.get("entities", 0)
    job = d.get("entity_job") or {}
    record("PASS" if ent else "WARN", "entities extracted",
           f"{ent} entities; entity job: {job.get('status', 'never queued')}")


# ---------------------------------------------------------------------------- Laravel API

def check_laravel(args, token, other_token):
    print("\nLaravel API")
    if args.doc_id is None:
        record("SKIP", "figure routes and ownership", "give --doc-id")
        return
    base = args.laravel_url.rstrip("/")
    auth = {"Authorization": f"Bearer {token}"} if token else {}

    code, _ = http("POST", f"{base}/api/graph-rag/callback", {})
    record("PASS" if code == 401 else "FAIL", "callback route rejects callers without the key",
           "" if code == 401 else f"HTTP {code}: anyone could mark documents finished (set GRAPH_RAG_API_KEY)")

    code, _ = http("GET", f"{base}/api/documents/{args.doc_id}/figures")
    record("PASS" if code == 401 else "FAIL", "figure list requires login", f"HTTP {code}")

    if not token:
        record("SKIP", "figure list / image as the owner", "set LARAVEL_TOKEN")
    else:
        code, body = http("GET", f"{base}/api/documents/{args.doc_id}/figures", headers=auth)
        if code != 200:
            record("FAIL", "owner can list figures", f"HTTP {code} {str(body)[:100]}")
        else:
            record("PASS", "owner can list figures", f"{body.get('total')} figure(s)")
            if body.get("total"):
                url = body["figures"][0]["image_url"]
                good_url = url.startswith("/api/documents/")
                record("PASS" if good_url else "FAIL", "image URLs point at this app, not the internal service", url)
                code, raw = http("GET", f"{base}{url}", headers=auth, binary=True)
                is_png = code == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
                record("PASS" if is_png else "FAIL", "image is a real PNG",
                       f"HTTP {code}, {len(raw) if isinstance(raw, bytes) else 0} bytes")
            else:
                record("SKIP", "image download", "this document has no figures")

    if not other_token:
        record("SKIP", "another user is refused", "set LARAVEL_TOKEN_OTHER (a different user's token)")
    else:
        code, _ = http("GET", f"{base}/api/documents/{args.doc_id}/figures",
                       headers={"Authorization": f"Bearer {other_token}"})
        record("PASS" if code in (403, 404) else "FAIL", "another user cannot read this document's figures",
               f"HTTP {code}" + ("" if code in (403, 404) else ": OWNERSHIP CHECK MISSING"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rag-url", default="http://localhost:8001")
    p.add_argument("--laravel-url", default="http://localhost:8000")
    p.add_argument("--doc-id", type=int, help="a processed document that has an owner and (ideally) a figure")
    p.add_argument("--user-id", default="1", help="owner of --doc-id")
    args = p.parse_args()

    key = os.getenv("INTERNAL_API_KEY")
    rag_headers = {"X-API-Key": key} if key else {}
    token, other = os.getenv("LARAVEL_TOKEN"), os.getenv("LARAVEL_TOKEN_OTHER")

    if check_python(args, rag_headers):
        check_verifier(args, rag_headers)
        check_document(args, rag_headers)
    check_laravel(args, token, other)

    counts = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['WARN']} warnings, {counts['SKIP']} skipped")
    if counts["FAIL"]:
        print("FAILED checks:")
        for status, name, detail in RESULTS:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
