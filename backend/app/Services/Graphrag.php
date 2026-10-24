<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Python document-processing service (FastAPI)
    |--------------------------------------------------------------------------
    | Reuses your existing PYTHON_PROCESSOR_URL.
    | api_key must equal INTERNAL_API_KEY in the Python service's .env. It is sent on
    | every request to Python AND required on the callback Python sends back to Laravel.
    */
    'base_url' => env('PYTHON_PROCESSOR_URL', 'http://localhost:8001'),
    'api_key' => env('GRAPH_RAG_API_KEY'),
    'timeout' => (int) env('GRAPH_RAG_TIMEOUT', 60),

    /*
    |--------------------------------------------------------------------------
    | Language model (Ollama)
    |--------------------------------------------------------------------------
    | Uses /api/chat (not /v1/chat/completions): only /api/chat honours `options`,
    | so num_ctx really applies. Too small a num_ctx silently truncates long prompts.
    */
    'ollama' => [
        'url' => env('OLLAMA_URL', 'http://localhost:11434'),
        'model' => env('OLLAMA_MODEL', 'qwen2.5:7b'),
        'timeout' => (int) env('OLLAMA_TIMEOUT', 280),
        'num_ctx' => (int) env('OLLAMA_NUM_CTX', 8192),
        'num_predict' => (int) env('OLLAMA_NUM_PREDICT', 768),
        'keep_alive' => env('OLLAMA_KEEP_ALIVE', '10m'),
    ],

    /*
    | Model per answer language. Small models (3B) often ignore the citation format;
    | try e.g. GRAPH_RAG_MODEL_EN=qwen2.5:7b (no .php change needed).
    */
    'models' => [
        'english' => env('GRAPH_RAG_MODEL_EN', 'llama3.2'),
        'persian' => env('GRAPH_RAG_MODEL_FA', 'gemma3:4b'),
    ],

    /*
    |--------------------------------------------------------------------------
    | Agent loop limits
    |--------------------------------------------------------------------------
    */
    'agent' => [
        'max_subquestions' => (int) env('GRAPH_RAG_MAX_SUBQUESTIONS', 4),
        'max_iterations' => (int) env('GRAPH_RAG_MAX_ITERATIONS', 2),
        'evidence_per_doc' => (int) env('GRAPH_RAG_EVIDENCE_PER_DOC', 6),
        'grade_snippet_chars' => 500,
        'history_answer_chars' => 300,
    ],

];