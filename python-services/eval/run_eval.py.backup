#!/usr/bin/env python3
"""
Evaluation harness (RFP section 7). Standard library only.

Two modes:

  retrieval   Fast, no LLM. Calls the Python service's POST /retrieve for each question and
              checks that the expected passage is among the evidence.
                python eval/run_eval.py retrieval --questions eval/questions.json --docs 3

  answer      End to end through the Laravel API: ask, wait for the queued job, then check
              the answer, its citations and the verifier's warnings.
                python eval/run_eval.py answer --questions eval/questions.json \
                       --laravel-url http://localhost:8000 --session 1

Secrets are read from the environment, never from arguments (so they don't land in shell
history or in anything you paste):  INTERNAL_API_KEY  (Python service),  LARAVEL_TOKEN.

Questions file: a JSON list. Every field except "q" is optional:
  {"q": "What is reference counting?",
   "expect": ["reference counting"],       passage must contain ANY of these (case-insensitive)
   "expect_pages": [14],                   ...or come from one of these pages
   "answer_contains": ["memory"],          answer must contain ANY of these (answer mode)
   "unanswerable": true,                   the documents do not cover it: a good system says so
   "document_ids": [3]}                    overrides --docs for this question
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

REFUSAL = re.compile(
    r"(not (contain|mention|provide|include|found|covered)|no (relevant )?information|"
    r"cannot (find|answer)|can't (find|answer)|couldn.?t find|could not find|"
    r"do(es)? not (say|specify|discuss|address))",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------- helpers

def http_json(method, url, payload=None, headers=None, timeout=120):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, {"raw": body[:300]}
    except Exception as e:  # connection refused, timeout, ...
        return 0, {"error": str(e)}


def norm(text):
    return " ".join((text or "").lower().split())


def contains_any(haystack, needles):
    h = norm(haystack)
    return any(norm(n) in h for n in needles)


def is_bibliography(headings, text):
    last = re.sub(r"^[\d.\s]+", "", (headings or [""])[-1] or "").strip()
    if re.match(r"^(?:references?|reference list|bibliography|works cited|literature cited|"
                r"منابع|مراجع|فهرست منابع|کتابنامه)\s*[:.]?$", last, re.I):
        return True
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    hits = len(re.findall(r"^\s*(?:[-*\u2022]\s*)?\[\d{1,3}\]\s", text or "", re.M))
    return len(lines) >= 2 and hits / len(lines) >= 0.6


def pct(part, whole):
    return f"{100.0 * part / whole:5.1f}%" if whole else "   n/a"


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list) or not all(isinstance(q, dict) and q.get("q") for q in questions):
        sys.exit("The questions file must be a JSON list of objects that each have a 'q' field.")
    return questions


# ------------------------------------------------------------------------- retrieval mode

def run_retrieval(args, questions):
    key = os.getenv("INTERNAL_API_KEY") or args.api_key
    headers = {"X-API-Key": key} if key else {}
    rows = []

    for q in questions:
        row = {"q": q["q"], "skipped": None}
        expect, pages = q.get("expect") or [], q.get("expect_pages") or []
        if q.get("unanswerable") or not (expect or pages):
            row["skipped"] = "unanswerable" if q.get("unanswerable") else "no expectation"
            rows.append(row)
            continue

        docs = q.get("document_ids") or args.docs
        t0 = time.time()
        status, body = http_json(
            "POST", f"{args.rag_url}/retrieve",
            {"document_ids": docs, "queries": [{"q": q["q"]}]}, headers,
        )
        row["seconds"] = round(time.time() - t0, 2)
        evidence = body.get("evidence", []) if status == 200 else []
        row["http"] = status
        row["evidence"] = len(evidence)

        ranked = sorted(evidence, key=lambda e: e.get("score", 0), reverse=True)
        row["rank"] = None
        for i, e in enumerate(ranked, start=1):
            blob = (e.get("text") or "") + " " + " ".join(e.get("headings") or [])
            if (expect and contains_any(blob, expect)) or (pages and e.get("page") in pages):
                row["rank"] = i
                break
        row["hit"] = row["rank"] is not None
        row["bibliography_chunks"] = sum(
            1 for e in evidence if is_bibliography(e.get("headings"), e.get("text"))
        )
        rows.append(row)

    scored = [r for r in rows if not r["skipped"]]
    hits = [r for r in scored if r["hit"]]
    summary = {
        "mode": "retrieval",
        "questions_scored": len(scored),
        "hit_rate": round(len(hits) / len(scored), 3) if scored else None,
        "mrr": round(sum(1.0 / r["rank"] for r in hits) / len(scored), 3) if scored else None,
        "avg_seconds": round(statistics.mean(r["seconds"] for r in scored), 2) if scored else None,
        "questions_with_reference_lists_in_evidence": sum(1 for r in scored if r["bibliography_chunks"]),
        "failed_requests": sum(1 for r in scored if r["http"] != 200),
    }
    print_retrieval(rows, summary)
    return rows, summary


def print_retrieval(rows, summary):
    print(f"\n{'#':>3}  {'result':8} {'rank':>4} {'refs':>4} {'sec':>6}  question")
    for i, r in enumerate(rows, start=1):
        if r["skipped"]:
            print(f"{i:>3}  {'skipped':8} {'':>4} {'':>4} {'':>6}  {r['q'][:70]}  ({r['skipped']})")
            continue
        verdict = "HIT" if r["hit"] else "MISS"
        print(f"{i:>3}  {verdict:8} {str(r['rank'] or '-'):>4} {r['bibliography_chunks']:>4} "
              f"{r['seconds']:>6}  {r['q'][:70]}")
    s = summary
    print("\n--- retrieval summary ---")
    print(f"questions scored        : {s['questions_scored']}")
    print(f"hit rate                : {s['hit_rate']}   (expected passage found in the evidence)")
    print(f"MRR                     : {s['mrr']}   (1.0 = always ranked first)")
    print(f"avg latency             : {s['avg_seconds']} s")
    print(f"with reference lists    : {s['questions_with_reference_lists_in_evidence']}   (should be 0)")
    if s["failed_requests"]:
        print(f"FAILED REQUESTS         : {s['failed_requests']}  (is the service up? is INTERNAL_API_KEY set?)")


# ----------------------------------------------------------------------------- answer mode

def run_answers(args, questions):
    token = os.getenv("LARAVEL_TOKEN") or args.token
    if not token:
        sys.exit("Set LARAVEL_TOKEN in the environment (a Sanctum token for the test user).")
    headers = {"Authorization": f"Bearer {token}"}
    base = args.laravel_url.rstrip("/")
    rows = []

    for q in questions:
        row = {"q": q["q"], "error": None}
        t0 = time.time()
        status, body = http_json(
            "POST", f"{base}/api/sessions/{args.session}/search",
            {args.question_field: q["q"]}, headers,
        )
        if status not in (200, 201, 202) or "question_id" not in body:
            row.update(error=f"submit failed: HTTP {status} {str(body)[:120]}", passed=False)
            rows.append(row)
            continue

        qid = body["question_id"]
        while time.time() - t0 < args.timeout:
            _, st = http_json("GET", f"{base}/api/questions/{qid}/status", None, headers)
            if st.get("is_completed"):
                break
            time.sleep(args.poll)
        row["seconds"] = round(time.time() - t0, 1)

        _, res = http_json("GET", f"{base}/api/questions/{qid}/result", None, headers)
        answer = res.get("answer") or ""
        sources = res.get("sources") or []
        warnings = res.get("warnings") or []
        source_text = " ".join(
            (s.get("text") or "") + " " + " ".join(s.get("headings") or []) for s in sources
        )

        row.update(
            completed=res.get("status") == "completed",
            cited=len(sources) > 0,
            clean=not warnings,
            warnings=warnings,
            answer=answer,
            n_sources=len(sources),
        )

        if q.get("unanswerable"):
            row["passed"] = row["completed"] and (not row["cited"] or bool(REFUSAL.search(answer)))
        else:
            expect, must = q.get("expect") or [], q.get("answer_contains") or []
            row["grounded"] = contains_any(answer + " " + source_text, expect) if expect else None
            row["answer_ok"] = contains_any(answer, must) if must else None
            row["passed"] = (
                row["completed"] and row["cited"] and row["clean"]
                and row["grounded"] is not False and row["answer_ok"] is not False
            )
        rows.append(row)

    summary = summarize_answers(rows)
    print_answers(rows, summary, show_answers=args.show_answers)
    return rows, summary


def summarize_answers(rows):
    ran = [r for r in rows if not r["error"]]
    times = sorted(r["seconds"] for r in ran if "seconds" in r)
    return {
        "mode": "answer",
        "questions": len(rows),
        "passed": sum(1 for r in rows if r.get("passed")),
        "completed": sum(1 for r in ran if r.get("completed")),
        "with_citations": sum(1 for r in ran if r.get("cited")),
        "clean_no_warnings": sum(1 for r in ran if r.get("clean")),
        "submit_errors": sum(1 for r in rows if r["error"]),
        "median_seconds": round(statistics.median(times), 1) if times else None,
        "p90_seconds": times[min(len(times) - 1, int(0.9 * len(times)))] if times else None,
    }


def print_answers(rows, summary, show_answers=False):
    print(f"\n{'#':>3}  {'result':6} {'cites':>5} {'warn':>4} {'sec':>6}  question")
    for i, r in enumerate(rows, start=1):
        if r["error"]:
            print(f"{i:>3}  {'ERROR':6} {'':>5} {'':>4} {'':>6}  {r['q'][:60]}  -> {r['error']}")
            continue
        print(f"{i:>3}  {'PASS' if r['passed'] else 'FAIL':6} {r['n_sources']:>5} "
              f"{len(r['warnings']):>4} {r['seconds']:>6}  {r['q'][:60]}")
        if show_answers or not r["passed"]:
            print(f"       answer  : {r['answer'][:220].replace(chr(10), ' ')}")
            for w in r["warnings"]:
                print(f"       warning : {w}")
    s, n = summary, summary["questions"]
    print("\n--- answer summary ---")
    print(f"passed                  : {s['passed']}/{n}  {pct(s['passed'], n)}")
    print(f"completed               : {s['completed']}/{n}")
    print(f"with citations          : {s['with_citations']}/{n}")
    print(f"clean (no warnings)     : {s['clean_no_warnings']}/{n}")
    print(f"latency median / p90    : {s['median_seconds']} s / {s['p90_seconds']} s")
    if s["submit_errors"]:
        print(f"SUBMIT ERRORS           : {s['submit_errors']}  (token, session id, or --question-field wrong?)")


# ---------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--questions", required=True, help="path to the questions JSON file")
        sp.add_argument("--out", help="write full results as JSON to this file")

    r = sub.add_parser("retrieval", help="retrieval quality, no LLM")
    common(r)
    r.add_argument("--rag-url", default="http://localhost:8001")
    r.add_argument("--docs", type=int, nargs="+", required=True, help="document ids to search")
    r.add_argument("--api-key", help="prefer the INTERNAL_API_KEY environment variable")

    a = sub.add_parser("answer", help="end to end through the Laravel API")
    common(a)
    a.add_argument("--laravel-url", default="http://localhost:8000")
    a.add_argument("--session", type=int, required=True, help="chat session that holds the documents")
    a.add_argument("--question-field", default="content", help="JSON field your search route expects")
    a.add_argument("--token", help="prefer the LARAVEL_TOKEN environment variable")
    a.add_argument("--timeout", type=int, default=900, help="seconds to wait per question")
    a.add_argument("--poll", type=float, default=3.0)
    a.add_argument("--show-answers", action="store_true")

    args = p.parse_args()
    questions = load_questions(args.questions)
    rows, summary = (run_retrieval if args.mode == "retrieval" else run_answers)(args, questions)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nfull results written to {args.out}")


if __name__ == "__main__":
    main()
