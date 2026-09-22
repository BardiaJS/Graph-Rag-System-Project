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

    public $tries = 3;

    public $backoff = [30, 60, 120];

    protected User $user;
    protected ChatSession $session;
    protected Question $question;
    protected array $documentIds;

    public function __construct(
        User $user,
        ChatSession $session,
        Question $question,
        array $documentIds
    ) {
        $this->user = $user;
        $this->session = $session;
        $this->question = $question;
        $this->documentIds = $documentIds;
    }

    public function handle(): void
    {
        Log::info('🔄 AgenticAnswerJob started', [
            'question_id' => $this->question->id,
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds,
        ]);

        try {

            Log::info('📤 Sending question to FastAPI', [
                'question_id' => $this->question->id,
                'document_ids' => $this->documentIds,
                'top_k' => 5,
            ]);

            $response = Http::timeout(120)
                ->post('http://localhost:8001/ask', [
                    'question' => $this->question->content,
                    'document_ids' => $this->documentIds,
                    'top_k' => 5,
                    'session_id' => $this->session->id,
                ]);

            Log::info('📥 Response received from FastAPI', [
                'question_id' => $this->question->id,
                'status' => $response->status(),
                'successful' => $response->successful(),
                'body' => $response->body(),
            ]);

            /*
             * FastAPI موفق
             */
            if ($response->successful()) {

                $data = $response->json();

                Log::info('📦 FastAPI JSON decoded', [
                    'question_id' => $this->question->id,
                    'data' => $data,
                ]);

                $answer = Answer::create([
                    'question_id' => $this->question->id,
                    'answer' => $data['answer'] ?? 'پاسخی یافت نشد',
                    'sources' => $data['sources'] ?? null,
                    'sub_queries' => $data['sub_queries'] ?? null,
                    'search_plans' => $data['search_plans'] ?? null,
                    'num_searches' => $data['num_searches'] ?? 0,
                    'version' => 1,
                    'is_latest' => true,
                    'answered_at' => now(),
                ]);

                $this->question->update([
                    'status' => 'completed',
                    'error_message' => null,
                ]);

                Log::info('✅ AgenticAnswerJob completed', [
                    'question_id' => $this->question->id,
                    'answer_id' => $answer->id,
                    'num_searches' => $data['num_searches'] ?? 0,
                ]);

                return;
            }

            /*
             * FastAPI خطا برگردانده
             */
            $errorBody = $response->body();

            Log::error('❌ FastAPI returned an error', [
                'question_id' => $this->question->id,
                'status' => $response->status(),
                'body' => $errorBody,
            ]);

            $this->question->update([
                'status' => 'failed',
                'error_message' => sprintf(
                    'Python service returned HTTP %s: %s',
                    $response->status(),
                    $errorBody
                ),
            ]);

            throw new \Exception(
                "FastAPI returned HTTP {$response->status()}: {$errorBody}"
            );

        } catch (\Throwable $e) {

            Log::error('❌ AgenticAnswerJob exception', [
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

    public function failed(\Throwable $exception): void
    {
        Log::error('💀 AgenticAnswerJob permanently failed', [
            'question_id' => $this->question->id,
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds,
            'error' => $exception->getMessage(),
        ]);

        $this->question->update([
            'status' => 'failed',
            'error_message' => $exception->getMessage(),
        ]);
    }
}
