<?php

namespace App\Jobs\Document;

use App\Models\ChatSession;
use App\Models\Document;
use App\Models\User;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Foundation\Queue\Queueable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class DocumentProcessJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    // تعداد دفعات تلاش مجدد در صورت خطا
    public $tries = 3;
    
    // زمان تاخیر بین تلاش‌های مجدد (بر حسب ثانیه)
    public $backoff = [30, 60, 120];

    protected $user;
    protected $session;
    protected $documentIds;

    public function __construct(User $user, ChatSession $session, array $documentIds)
    {
        $this->user = $user;
        $this->session = $session;
        $this->documentIds = $documentIds;
    }

    public function handle()
    {
        $pythonServiceUrl = config('services.python_processor.url', 'http://localhost:8001');
        
        Log::info('Starting document processing job', [
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds
        ]);

        $successCount = 0;
        $failedCount = 0;

        foreach ($this->documentIds as $docId) {
            $document = Document::find($docId);
            
            if (!$document) {
                Log::warning("Document {$docId} not found");
                $failedCount++;
                continue;
            }

            try {
                $response = Http::timeout(300) // 5 دقیقه تایم‌اوت
                    ->post("{$pythonServiceUrl}/process", [
                        'user_id' => $this->user->id,
                        'session_id' => $this->session->id,
                        'file_path' => $document->file_path,
                        'document_ids' => [$document->id]
                    ]);
                
                if ($response->successful()) {
                    Log::info("Document {$docId} sent to processing service successfully", [
                        'response' => $response->json()
                    ]);

                    $document->update([
                        'processing_status' => 'completed',
                        'processed_at' => now(),
                        'processing_error' => null,
                    ]);
                    $successCount++;
                } else {
                    Log::error("Failed to send document {$docId}", [
                        'status' => $response->status(),
                        'response' => $response->body()
                    ]);
                    $failedCount++;
                }
                
            } catch (\Exception $e) {
                Log::error("Exception while sending document {$docId}", [
                    'error' => $e->getMessage(),
                    'trace' => $e->getTraceAsString()
                ]);
                $failedCount++;
                
                // اگر خطا داشت، Job را دوباره به صف برگردان
                if ($this->attempts() < $this->tries) {
                    $this->release($this->backoff[$this->attempts() - 1] ?? 60);
                    return;
                }
            }
        }

        Log::info('Document processing job completed', [
            'total' => count($this->documentIds),
            'success' => $successCount,
            'failed' => $failedCount
        ]);

        // در صورت عدم موفقیت، خطا را ثبت کن
        if ($failedCount > 0 && $successCount === 0) {
            throw new \Exception("All documents failed to process");
        }




        Log::info('DOCUMENT JOB: before Python', [
            'document_id' => $docId,
        ]);

        $response = Http::timeout(300)
            ->post("{$pythonServiceUrl}/process", [
                'user_id' => $this->user->id,
                'session_id' => $this->session->id,
                'file_path' => $document->file_path,
                'document_ids' => [$document->id],
            ]);

        Log::info('DOCUMENT JOB: after Python', [
            'document_id' => $docId,
            'status' => $response->status(),
            'successful' => $response->successful(),
        ]);

        if ($response->successful()) {

            Log::info('DOCUMENT JOB: before database update', [
                'document_id' => $docId,
            ]);

            $document->update([
                'processing_status' => 'completed',
                'processed_at' => now(),
                'processing_error' => null,
            ]);

            Log::info('DOCUMENT JOB: after database update', [
                'document_id' => $docId,
                'status' => $document->fresh()->processing_status,
            ]);

            $successCount++;
        }
    }

    /**
     * اگر Job با خطا مواجه شود، این متد فراخوانی می‌شود
     */
    public function failed(\Throwable $exception)
    {
        Log::error('Document processing job failed permanently', [
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds,
            'error' => $exception->getMessage()
        ]);
    }
}