<?php

namespace App\Jobs\Rag;

use App\Models\Answer;
use App\Models\ChatSession;
use App\Models\Question;
use App\Models\User;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Foundation\Queue\Queueable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class AgenticAnswerJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 1;
    public $timeout = 350;

    protected $user;
    protected $session;
    protected $question;
    protected $documentIds;

    public function __construct(User $user, ChatSession $session, Question $question, array $documentIds)
    {
        $this->user = $user;
        $this->session = $session;
        $this->question = $question;
        $this->documentIds = $documentIds;
    }



/**
 * سوال رو به sub-queryها تقسیم کن (Agentic RAG)
 */
    protected function planQueries(string $question, string $ollamaUrl, string $ollamaModel): array
    {
        $prompt = "You are a query planner. Break down the user's question into 1-3 sub-queries " .
                "that can be searched independently.\n\n" .
                "User question: {$question}\n\n" .
                "Return ONLY a JSON object like this (no markdown, no explanation):\n" .
                '{"sub_queries": ["query 1", "query 2"], "reasoning": "why"}' . "\n\n" .
                "If the question is simple, return only 1 sub-query.";

        $payload = [
            'model' => $ollamaModel,
            'messages' => [
                ['role' => 'user', 'content' => $prompt],
            ],
            'temperature' => 0.1,
            'options' => [
                'num_ctx' => 4096,
                'num_predict' => 256,
            ],
            'keep_alive' => '10m',  

        ];

        $json = json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_UNESCAPED_UNICODE);

        try {
            $response = Http::timeout(60)
                ->connectTimeout(10)
                ->withHeaders(['Content-Type' => 'application/json'])
                ->withBody($json, 'application/json')
                ->post("{$ollamaUrl}/v1/chat/completions");

            $content = $response->json('choices.0.message.content') ?? '';

            Log::info('AgenticAnswerJob: plan response', [
                'content' => substr($content, 0, 300),
            ]);

            // استخراج JSON از پاسخ
            if (preg_match('/\{.*\}/s', $content, $matches)) {
                $parsed = json_decode($matches[0], true);
                if (isset($parsed['sub_queries']) && is_array($parsed['sub_queries'])) {
                    $subQueries = array_values(array_filter($parsed['sub_queries'], 'is_string'));
                    if (!empty($subQueries)) {
                        return [
                            'sub_queries' => $subQueries,
                            'reasoning' => $parsed['reasoning'] ?? '',
                        ];
                    }
                }
            }
        } catch (\Exception $e) {
            Log::warning('AgenticAnswerJob: planQueries failed', [
                'error' => $e->getMessage(),
            ]);
        }

        // Fallback: خود سوال
        return [
            'sub_queries' => [$question],
            'reasoning' => 'fallback',
        ];
    }





    public function handle()
    {
        $pythonServiceUrl = config('services.python_processor.url', 'http://localhost:8001');
        $ollamaUrl = config('services.ollama.url', 'http://localhost:11434');

            // ⭐ تشخیص زبان سوال
        $isPersian = preg_match('/[\x{0600}-\x{06FF}]/u', $this->question->content);
            
        if ($isPersian) {
            $ollamaModel = 'qwen2.5:7b';
            $answerLanguage = 'Persian (فارسی)';
            Log::info('AgenticAnswerJob: Persian detected, using qwen2.5:7b');
        } else {
            $ollamaModel = 'llama3.2';
            $answerLanguage = 'English';
            Log::info('AgenticAnswerJob: English detected, using llama3.2');
        }

        try {
            // ⭐ ۱. تاریخچه (سوال + جواب)
            $history = Question::where('chat_session_id', $this->session->id)
                ->where('id', '!=', $this->question->id)
                ->where('status', 'completed')
                ->with(['answers' => function ($q) {
                    $q->where('is_latest', true);
                }])
                ->orderBy('created_at', 'desc')
                ->limit(3)
                ->get()
                ->reverse()
                ->map(function ($q) {
                    $answer = $q->answers->first();
                    return [
                        'question' => $this->cleanUtf8($q->content),
                        'answer' => $answer ? $this->cleanUtf8($answer->content) : '',
                    ];
                })
                ->toArray();

            Log::info('AgenticAnswerJob: history loaded', [
                'count' => count($history),
            ]);

            // ⭐ ۲. Plan queries (Agentic RAG)
            $plan = $this->planQueries(
                $this->cleanUtf8($this->question->content),
                $ollamaUrl,
                $ollamaModel
            );

            $subQueries = $plan['sub_queries'];
            $reasoning = $plan['reasoning'];

            Log::info('AgenticAnswerJob: plan created', [
                'q_id' => $this->question->id,
                'sub_queries' => $subQueries,
                'reasoning' => $reasoning,
            ]);

            // ⭐ ۳. برای هر sub-query، search جدا
            $allResults = [];
            $seenIds = [];

            foreach ($subQueries as $sq) {
                $searchPayload = [
                    'query' => $this->cleanUtf8($sq),
                    'limit' => 2,
                    'document_id' => (int) $this->documentIds[0],
                ];

                $searchJson = json_encode($searchPayload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_UNESCAPED_UNICODE);

                $searchResponse = Http::timeout(30)
                    ->withHeaders(['Content-Type' => 'application/json'])
                    ->withBody($searchJson, 'application/json')
                    ->post("{$pythonServiceUrl}/search");

                if (!$searchResponse->successful()) {
                    Log::warning('AgenticAnswerJob: sub-search failed', [
                        'sub_query' => $sq,
                        'status' => $searchResponse->status(),
                    ]);
                    continue;
                }

                $subResults = $searchResponse->json('results') ?? [];

                foreach ($subResults as $r) {
                    $key = ($r['document_id'] ?? '') . '_' . ($r['chunk_index'] ?? '');
                    if (!isset($seenIds[$key])) {
                        $allResults[] = $r;
                        $seenIds[$key] = true;
                    }
                }
            }

            $results = $allResults;

            Log::info('AgenticAnswerJob: search done', [
                'q_id' => $this->question->id,
                'sub_queries_count' => count($subQueries),
                'total_results' => count($results),
            ]);
            // ⭐ ۳. Context
            $context = collect($results)
                ->take(5)
                ->map(fn($r, $i) => "[" . ($i + 1) . "] " . $this->cleanUtf8(substr($r['text'], 0, 200)))
                ->implode("\n\n");

            // ⭐ ۴. تاریخچه به متن
            $historyText = "";
            if (!empty($history)) {
                $historyText = "Previous conversation:\n";
                foreach ($history as $h) {
                    $historyText .= "User: {$h['question']}\n";
                    if (!empty($h['answer'])) {
                        $historyText .= "Assistant: " . substr($h['answer'], 0, 200) . "\n";
                    }
                    $historyText .= "\n";
                }
                $historyText .= "---\n\n";
            }

            // ⭐ ۵. Prompt
            $prompt = $historyText .
                "You are a helpful assistant. Answer the current question based on the context and previous conversation.\n\n" .
                "Context:\n{$context}\n\n" .
                "Current question: " . $this->cleanUtf8($this->question->content) . "\n\n" .
                "Answer in {$answerLanguage}, based only on the context above. " .
                "Keep English names (like 'Rasmus Lerdorf') and code terms (like 'array') in English.\n" .
                "Be concise and complete.";

            // ⭐ ۶. پاکسازی prompt
            $prompt = $this->cleanUtf8($prompt);

            Log::info('AgenticAnswerJob: before LLM', [
                'q_id' => $this->question->id,
                'prompt_len' => strlen($prompt),
                'history_len' => strlen($historyText),
                'context_len' => strlen($context),
                'context_results_count' => count($results),
            ]);

            // ⭐ ۷. LLM (با json_encode + withBody)
            $llmPayload = [
                'model' => $ollamaModel,
                'messages' => [
                    [
                        'role' => 'user',
                        'content' => $prompt,
                    ],
                ],
                'temperature' => 0.1,
                'options' => [
                    'num_ctx' => 8192,
                    'num_predict' => 512,
                ],
                'keep_alive' => '10m',  

            ];

            $llmJson = json_encode($llmPayload, JSON_INVALID_UTF8_SUBSTITUTE | JSON_UNESCAPED_UNICODE);

            Log::info('AgenticAnswerJob: llm payload', [
                'q_id' => $this->question->id,
                'json_error' => json_last_error_msg(),
                'json_len' => strlen($llmJson),
            ]);

            if ($llmJson === false) {
                throw new \Exception("Failed to encode LLM payload: " . json_last_error_msg());
            }

            $llmResponse = Http::timeout(280)
                ->connectTimeout(10)
                ->withHeaders(['Content-Type' => 'application/json'])
                ->withBody($llmJson, 'application/json')
                ->post("{$ollamaUrl}/v1/chat/completions");

            if (!$llmResponse->successful()) {
                throw new \Exception("LLM failed: " . $llmResponse->body());
            }

            $answer = $llmResponse->json('choices.0.message.content');

            Log::info('AgenticAnswerJob: after LLM', [
                'q_id' => $this->question->id,
                'answer_len' => strlen($answer),
            ]);

            // ⭐ ۸. Sources
            $sources = collect($results)->map(fn($r) => [
                'text' => $this->cleanUtf8($r['text']),
                'headings' => $r['headings'] ?? [],
                'page' => $r['page'] ?? null,
                'score' => $r['score'] ?? 0,
                'chunk_index' => $r['chunk_index'] ?? null,
                'source' => $r['source'] ?? 'vector',
            ])->toArray();

            // ⭐ ۹. ذخیره
            $this->saveAnswer(
                $this->cleanUtf8($answer),
                $sources,
                $subQueries,
                [$reasoning]
            );

        } catch (\Exception $e) {
            Log::error('AgenticAnswerJob failed', [
                'question_id' => $this->question->id,
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString(),
            ]);

            $this->question->update([
                'status' => 'failed',
                'error_message' => $e->getMessage(),
            ]);

            throw $e;
        }
    }

    /**
     * حذف کاراکترهای غیر UTF-8
     */
    protected function cleanUtf8(?string $text): string
    {
        if ($text === null) {
            return '';
        }

        // ۱. حذف BOM
        $text = preg_replace('/^\x{FEFF}/u', '', $text);

        // ۲. حذف کاراکترهای کنترلی (به جز \n \r \t)
        $text = preg_replace('/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/u', '', $text);

        // ۳. حذف byteهای ناقص UTF-8 (معتبر)
        if (!mb_check_encoding($text, 'UTF-8')) {
            $text = @iconv('UTF-8', 'UTF-8//IGNORE', $text);
        }

        // ۴. حذف کاراکترهای ناقص باقی‌مونده
        $text = preg_replace('/<[0-9A-F]{2}>/i', '', $text);
        $text = preg_replace('/\\\\uD[89AB][0-9A-F]{2}/i', '', $text);

        // ۵. اطمینان نهایی
        if (!mb_check_encoding($text, 'UTF-8')) {
            $text = mb_convert_encoding($text, 'UTF-8', 'UTF-8');
        }

        return $text;
    }

    /**
     * ذخیره جواب
     */
    protected function saveAnswer(string $content, array $sources, array $subQueries = [], array $searchPlans = []): void
    {
        Answer::create([
            'question_id' => $this->question->id,
            'content' => $content,
            'sources' => $sources,
            'sub_queries' => $subQueries,
            'search_plans' => $searchPlans,
            'num_searches' => count($sources),
            'version' => 1,
            'is_latest' => true,
            'answered_at' => now(),
        ]);

        $this->question->update([
            'status' => 'completed',
        ]);

        Log::info('AgenticAnswerJob: answer saved', [
            'question_id' => $this->question->id,
        ]);
    }

    /**
     * اگه job fail بشه
     */
    public function failed(\Throwable $exception): void
    {
        $this->question->update([
            'status' => 'failed',
            'error_message' => $exception->getMessage(),
        ]);
    }
}