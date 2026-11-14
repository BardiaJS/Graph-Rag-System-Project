# Graph RAG: document intelligence backend

Upload PDFs, ask questions across them, get answers with verified citations and figures.
Everything runs locally (Ollama): no document text leaves your infrastructure.

## Architecture

```
 Laravel (auth, sessions, questions, the LLM agent loop)
     |  HTTP + X-API-Key                    ^ callback (document finished)
     v                                      |
 Python service (FastAPI)  ----------------+
   ingestion   PDF -> OCR check -> Docling -> chunks -> figures -> Qdrant + Neo4j
   retrieval   vector + neighbour chunks + shared-entity hop + linked figures
   verify      citations: drops unknown ones, strips them from refusals, prunes citation stuffing;
               flags invented figure numbers and numbers that appear in no source
   documents   list / detail / delete across every store
     |
     +-- Redis (RQ queues) -- Qdrant (vectors) -- Neo4j (graph) -- Ollama (LLM)
```

**Division of labour.** Python does everything deterministic and document-aware.
Laravel owns the LLM loop (plan -> retrieve -> grade -> answer -> verify), users and sessions.
Python's `/ask` (the same loop server-side) is a disabled-by-default reference copy.

## What runs (exactly ONE of each)

| # | Process | Command (from its project root) |
|---|---|---|
| 1 | Python API | `uvicorn app.main:app --port 8001` |
| 2 | Ingestion worker | `python -m app.worker` |
| 3 | Entity worker (optional) | `python -m app.worker --entities` |
| 4 | Laravel app | `php artisan serve` |
| 5 | Laravel queue | `php artisan queue:work --timeout=900` |

Plus Redis, Qdrant, Neo4j and Ollama. Use Ctrl+C to stop, never Ctrl+Z (suspended workers
keep holding queue slots). After copying new code **restart every process**: a long-running
worker keeps the code it started with. `GET /health` reports how many workers are running and
warns when the count is wrong.

## Settings

**Python `.env`**

| Variable | Meaning |
|---|---|
| `INTERNAL_API_KEY` | Shared secret; must equal Laravel's `GRAPH_RAG_API_KEY` |
| `LARAVEL_CALLBACK_URL` | e.g. `http://localhost:8000/api/graph-rag/callback`. Unset = Laravel never learns a document finished |
| `LARAVEL_STORAGE_PATH` | Laravel's `storage/app/public` |
| `OLLAMA_MODEL` | A model that is actually installed (`ollama list`). Default `llama3.1:8b` usually is not |
| `OLLAMA_NUM_CTX` | Keep equal to Laravel's `OLLAMA_NUM_CTX` (8192); different values make Ollama reload the model |
| `ENABLE_ENTITY_EXTRACTION` | `1` (default) queues entity extraction after ingestion |
| `ENTITY_MODEL` | Optional smaller model for entities only, e.g. `qwen2.5:3b` |
| `DOCLING_OCR` | `auto` (default: OCR only if the PDF has no text layer), `on`, `off` |
| `DEBUG_PRINTS` | `0` silences the verbose chunk/Qdrant/Neo4j printouts |

**Laravel `.env`**

| Variable | Meaning |
|---|---|
| `GRAPH_RAG_API_KEY` | Same value as `INTERNAL_API_KEY` |
| `PYTHON_PROCESSOR_URL` | e.g. `http://localhost:8001` |
| `OLLAMA_URL`, `OLLAMA_NUM_CTX` | Ollama address; context size (8192) |
| `GRAPH_RAG_MODEL_EN` / `_FA` | Model per answer language. A 7B model (`qwen2.5:7b`) follows the citation format far better than a 3B one |
| `GRAPH_RAG_MAX_ITERATIONS` | Retrieve/grade rounds per question (default 2) |
| `GRAPH_RAG_FAST_PATH` | `true` (default): simple questions (short, no comparison, no earlier conversation) skip the planner and grader calls, so one model generation instead of three. Comparisons and follow-ups always take the full path |
| `OLLAMA_NUM_PREDICT` | Max answer tokens (default 512). Generation time grows with answer length |
| `GRAPH_RAG_MAX_SOURCES`, `GRAPH_RAG_EVIDENCE_CHAR_BUDGET` | Hard cap on sources per prompt (12 / 12000 chars). Keeps the prompt inside `num_ctx` and answers fast; every document keeps its best chunk |
| `APP_TIMEZONE` | Set it (e.g. `Asia/Tehran`): Laravel logs UTC by default while the Python logs use local time, which makes cross-checking confusing |

Also set `retry_after` in `config/queue.php` above 900 (the question job's timeout), and run
`php artisan config:clear` after any `.env` change. Never paste keys or tokens into chat or logs.

## Life of a document

1. Laravel stores the PDF and dispatches `DocumentProcessJob` -> `POST /process` (returns `202 {job_id}` at once).
2. Python worker: OCR check -> Docling conversion -> chunking -> figure images -> Qdrant vectors -> Neo4j graph. About a minute for 30 pages.
3. Python calls Laravel's callback: the document becomes `completed` and is searchable.
4. Entity extraction runs afterwards on its own queue (tens of minutes). It enriches the graph; nothing waits for it.
   Status: `GET /documents/{id}` -> `entity_job`.

## Python API (all need `X-API-Key`)

| Endpoint | Purpose |
|---|---|
| `POST /process` | Queue a PDF (`user_id`, `session_id`, `file_path`, `document_ids`) |
| `GET /jobs/{job_id}` | Ingestion job status, stage, result |
| `POST /retrieve` | Evidence for queries, restricted to the caller's `document_ids` |
| `POST /verify` | Check an answer's citations and figure references |
| `GET /documents`, `/documents/{id}` | List / detail (sizes, consistency, job and entity-job status) |
| `DELETE /documents/{id}` | Remove from Qdrant, Neo4j, disk (stops entity work; 409 while ingesting) |
| `POST /documents/{id}/entities` | (Re)queue entity extraction without re-processing the PDF |
| `GET /documents/{id}/figures[/{n}/image]` | Figure metadata and PNGs |
| `POST /search` | Raw vector search (no LLM) |
| `GET /health` | Redis, queues, workers, Qdrant, Neo4j, Ollama; `status` + `warnings` |

Pass `user_id` on user-facing calls. Another user's document answers 404, not 403.

## Evaluation (RFP section 7)

```bash
export INTERNAL_API_KEY=...        # read from your .env; never type it into a command
python eval/run_eval.py retrieval --questions eval/questions.sample.json --docs 3

export LARAVEL_TOKEN=...           # a Sanctum token for a test user
python eval/run_eval.py answer --questions eval/questions.sample.json --session 1 --show-answers
```

`retrieval` is fast and needs no LLM: hit rate, MRR, reference-list contamination.
`answer` goes through your real API: pass rate, citation rate, warning-free rate, latency, average
sources cited per answer, a per-step time breakdown (plan / retrieve / grade / answer LLM / verify, from
the stored trace), and whether unanswerable questions are refused WITHOUT citing sources (a refusal that
cites many sources fails). A 100% pass rate is only as meaningful as the questions: read some answers
(`--show-answers`), do not trust the number alone. Write your own question file
(see the docstring at the top of `run_eval.py`); re-run after every prompt, model or chunking change.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Document stays `processing` | `LARAVEL_CALLBACK_URL` unset or wrong, or the key differs (401 in `laravel.log`) |
| Callback payload misses fields, `qdrant_points: 0` | An old worker is still running. `ps aux \| grep app.worker`, `pkill -9 -f app.worker`, start one |
| `LLM call failed ... 404` | `OLLAMA_MODEL` is not installed. The worker prints the installed names at startup |
| Entity extraction stops with "model unavailable" | Ollama down or wrong model (3 failures in a row abort it). Fix, then `POST /documents/{id}/entities` |
| An answer cites nearly every source (e.g. 12 of 12), even a refusal | Over-citing. A refusal ("the sources do not cover this") correctly cites nothing and is no longer sent through the citation repair pass. The eval's `avg cited sources` and `unanswerable handled` lines track this |
| Answer has no `[S#]` citations | Small model. Use a 7B model; the loop already retries once |
| `cURL error 7 ... port 8001` in `laravel.log` | The Python API was not running. Laravel retries for ~15 s (covers a restart); otherwise start uvicorn and re-ask. `GET /health` shows what is up |
| Question takes minutes | Many retrieve/grade rounds on a slow model: lower `GRAPH_RAG_MAX_ITERATIONS`, use a faster model; check `ollama ps` for GPU use |
| `Class ... not found` after copying files | Namespace/path mismatch (this project uses flat `App\Services`); run `composer dump-autoload` |
| Question job runs twice / ends `failed` | Queue `retry_after` is below the job timeout |
| `/health` says "N ingestion workers" | Stale workers; kill all, start one |

## Known limits

- Entity extraction is slow on CPU; use a smaller `ENTITY_MODEL` or a GPU.
- Section nodes are keyed by heading path, so repeated headings ("Example:") merge into one section node (retrieval is unaffected).
- Figures are found through captions and "Figure N" mentions; there is no image understanding, and appendix labels such as "Figure A1" are not parsed.
- Tables are not extracted as structured data (`do_table_structure` is off for speed).
- OCR quality is not scored; Persian scans depend on the OCR engine's language models being installed offline.
- Laravel's `metadata.entities` stays 0 (no second callback); read entity counts from `GET /documents/{id}`.
