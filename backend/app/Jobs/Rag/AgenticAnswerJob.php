<?php

namespace App\Jobs\Rag;

use App\Models\Answer;
use App\Models\ChatSession;
use App\Models\Question;
use App\Models\User;
use App\Services\AgentService;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Foundation\Queue\Queueable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;

class AgenticAnswerJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 1;

    // The loop can make several LLM calls. IMPORTANT: your queue connection's
    // `retry_after` (config/queue.php) must be LARGER than this, and `queue:work`
    // must run with --timeout larger than this, otherwise the job is picked up twice.
    public $timeout = 900;

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

    public function handle(AgentService $agent)
    {
        // ⭐ تشخیص زبان سوال
        $isPersian = (bool) preg_match('/[\x{0600}-\x{06FF}]/u', $this->question->content);

        if ($isPersian) {
            $ollamaModel = (string) config('graphrag.models.persian', 'gemma3:4b');   // چندزبانه، فارسی خوب
            $answerLanguage = 'Persian (فارسی)';
            $extraInstructions = 'DO NOT use Chinese characters (汉字). '
                . "DO NOT use English except for names like 'Rasmus Lerdorf' and code like 'array'.";
            Log::info('AgenticAnswerJob: Persian detected', ['model' => $ollamaModel]);
        } else {
            $ollamaModel = (string) config('graphrag.models.english', 'llama3.2');
            $answerLanguage = 'English';
            $extraInstructions = null;
            Log::info('AgenticAnswerJob: English detected', ['model' => $ollamaModel]);
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
                ->values()
                ->toArray();

            Log::info('AgenticAnswerJob: history loaded', ['count' => count($history)]);

            // ⭐ ۲. Plan → retrieve (all documents) → grade → answer → verify
            $result = $agent->answer(
                $this->cleanUtf8($this->question->content),
                $this->documentIds,
                [
                    'model' => $ollamaModel,
                    'history' => $history,
                    'answer_language' => $answerLanguage,
                    'extra_instructions' => $extraInstructions,
                ]
            );

            Log::info('AgenticAnswerJob: answer ready', [
                'q_id' => $this->question->id,
                'citations' => count($result['citations']),
                'warnings' => $result['warnings'],
            ]);

            // ⭐ ۳. Sources (old keys kept for the frontend, new keys added)
            $sources = array_map(fn ($c) => [
                'text' => $this->cleanUtf8($c['quote'] ?? ''),
                'headings' => $c['headings'] ?? [],
                'page' => $c['page'] ?? null,
                'score' => $c['score'] ?? 0,
                'chunk_index' => $c['chunk_index'] ?? null,
                'source' => !empty($c['linked_via']) ? 'graph' : 'vector',
                // new
                'id' => $c['id'] ?? null,
                'document_id' => $c['document_id'] ?? null,
                'figures' => $c['figures'] ?? [],
                'linked_via' => $c['linked_via'] ?? [],
            ], $result['citations']);

            $subQueries = [];
            $searches = 0;
            foreach ($result['trace'] as $step) {
                if (($step['step'] ?? null) === 'plan') {
                    $subQueries = array_map(fn ($s) => $s['q'], $step['sub_questions'] ?? []);
                }
                if (($step['step'] ?? null) === 'retrieve') {
                    $searches++;
                }
            }

            // ⭐ ۴. ذخیره
            $this->saveAnswer(
                $this->cleanUtf8($result['answer']),
                $sources,
                $subQueries,
                ['trace' => $result['trace'], 'warnings' => $result['warnings']],
                $searches
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

        // ۳. اگه UTF-8 معتبر نبود، با iconv تمیز کن
        if (!mb_check_encoding($text, 'UTF-8')) {
            $text = @iconv('UTF-8', 'UTF-8//IGNORE', $text);
        }

        // ۴. حذف byteهای ناقص (اینا واقعاً مشکل‌دارن)
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
    protected function saveAnswer(
        string $content,
        array $sources,
        array $subQueries = [],
        array $searchPlans = [],
        int $numSearches = 0
    ): void {
        Answer::create([
            'question_id' => $this->question->id,
            'content' => $content,
            'sources' => $sources,
            'sub_queries' => $subQueries,
            'search_plans' => $searchPlans,
            'num_searches' => $numSearches,
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

    public function failed(\Throwable $exception): void
    {
        $this->question->update([
            'status' => 'failed',
            'error_message' => $exception->getMessage(),
        ]);
    }
}
