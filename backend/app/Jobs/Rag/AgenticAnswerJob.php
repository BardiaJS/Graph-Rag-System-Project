<?php

namespace App\Jobs\Rag;

use App\Models\ChatSession;
use App\Models\Question;
use App\Models\Answer;  // ← این را اضافه کنید
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

    public function __construct(User $user, ChatSession $session, Question $question, array $documentIds)
    {
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
            'session_id' => $this->session->id
        ]);

        try {
            $response = Http::timeout(120)
                ->post('http://localhost:8001/ask', [
                    'question' => $this->question->content,
                    'document_ids' => $this->documentIds,
                    'top_k' => 5,
                    'session_id' => $this->session->id
                ]);

            if ($response->successful()) {
                $data = $response->json();
                
                Log::info('📥 Data received from FastAPI', [
                    'has_answer' => isset($data['answer']),
                    'answer_preview' => substr($data['answer'] ?? '', 0, 100)
                ]);

                // 1. ایجاد Answer جدید
                $answer = Answer::create([
                    'question_id' => $this->question->id,
                    'answer' => $data['answer'] ?? 'پاسخی یافت نشد',
                    'sources' => $data['sources'] ?? null,
                    'sub_queries' => $data['sub_queries'] ?? null,
                    'search_plans' => $data['search_plans'] ?? null,
                    'num_searches' => $data['num_searches'] ?? 0,
                    'version' => 1,
                    'is_latest' => true,
                    'answered_at' => now()
                ]);

                // 2. به‌روزرسانی Question
                $this->question->update([
                    'status' => 'completed'
                ]);

                Log::info('✅ AgenticAnswerJob completed', [
                    'question_id' => $this->question->id,
                    'answer_id' => $answer->id,
                    'num_searches' => $data['num_searches'] ?? 0
                ]);

            } else {
                $this->question->update([
                    'status' => 'failed',
                    'error_message' => 'HTTP Error: ' . $response->status()
                ]);

                Log::error('❌ AgenticAnswerJob failed', [
                    'question_id' => $this->question->id,
                    'status' => $response->status()
                ]);

                throw new \Exception('Failed to get answer from Python service');
            }

        } catch (\Exception $e) {
            Log::error('❌ AgenticAnswerJob exception', [
                'question_id' => $this->question->id,
                'error' => $e->getMessage()
            ]);

            $this->question->update([
                'status' => 'failed',
                'error_message' => $e->getMessage()
            ]);

            throw $e;
        }
    }

    public function failed(\Throwable $exception): void
    {
        Log::error('💀 AgenticAnswerJob permanently failed', [
            'question_id' => $this->question->id,
            'error' => $exception->getMessage()
        ]);

        $this->question->update([
            'status' => 'failed',
            'error_message' => $exception->getMessage()
        ]);
    }
}