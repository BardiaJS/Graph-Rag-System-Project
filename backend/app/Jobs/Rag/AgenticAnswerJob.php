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
    public $timeout = 300;

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

    public function handle()
    {
        $pythonServiceUrl = config('services.python_processor.url', 'http://localhost:8001');
        $ollamaUrl = config('services.ollama.url', 'http://localhost:11434');
        $ollamaModel = config('services.ollama.model', 'llama3.2');

        try {
            // ۱. Search
            $searchResponse = Http::timeout(30)
                ->post("{$pythonServiceUrl}/search", [
                    'query' => $this->question->content,
                    'limit' => 3, 
                    'document_id' => (int) $this->documentIds[0],
                ]);

            if (!$searchResponse->successful()) {
                throw new \Exception("Search failed: " . $searchResponse->body());
            }

            $results = $searchResponse->json('results') ?? [];

            if (empty($results)) {
                $this->saveAnswer("متأسفانه اطلاعاتی برای پاسخ به این سوال پیدا نشد.", []);
                return;
            }

            // ۲. Context (truncate شده)
            $context = collect($results)
                ->map(fn($r, $i) => "[" . ($i + 1) . "] " . substr($r['text'], 0, 400))
                ->implode("\n\n");

            // ۳. LLM
            $llmResponse = Http::timeout(120)
                ->post("{$ollamaUrl}/v1/chat/completions", [
                    'model' => $ollamaModel,
                    'messages' => [
                        [
                            'role' => 'user',
                            'content' => "بر اساس context زیر جواب بده. فقط از context استفاده کن.\n\n" .
                                        "Context:\n{$context}\n\n" .
                                        "سوال: {$this->question->content}",
                        ],
                    ],
                    'temperature' => 0.1,   // ← کمتر
                ]);

            if (!$llmResponse->successful()) {
                throw new \Exception("LLM failed: " . $llmResponse->body());
            }

            $answer = $llmResponse->json('choices.0.message.content');

            // ۴. ذخیره
            $sources = collect($results)->map(fn($r) => [
                'text' => $r['text'],
                'headings' => $r['headings'] ?? [],
                'page' => $r['page'] ?? null,
                'score' => $r['score'] ?? 0,
                'chunk_index' => $r['chunk_index'] ?? null,
                'source' => $r['source'] ?? 'vector',
            ])->toArray();

            $this->saveAnswer($answer, $sources);

        } 
        catch (\Exception $e) {
            Log::error('AgenticAnswerJob failed', [
                'question_id' => $this->question->id,
                'error' => $e->getMessage(),
            ]);

            $this->question->update([
                'status' => 'failed',
                'error_message' => $e->getMessage(),
            ]);

            throw $e;
        }
    }













    protected function saveAnswer(string $content, array $sources)
    {
        Answer::create([
            'question_id' => $this->question->id,
            'content' => $content,
            'sources' => $sources,
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

    public function failed(\Throwable $exception)
    {
        $this->question->update([
            'status' => 'failed',
            'error_message' => $exception->getMessage(),
        ]);
    }
}