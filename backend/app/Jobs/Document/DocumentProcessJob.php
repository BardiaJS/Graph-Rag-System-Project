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

    // تعداد تلاش‌ها رو کم کن (چون Docling سنگینه)
    public $tries = 1;
    
    // timeout خود job (کمتر از timeout HTTP باشه)
    public $timeout = 240;  // ۴ دقیقه

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
                Log::info('DOCUMENT JOB: before Python', ['document_id' => $docId]);

                // timeout رو کم کن: ۱۸۰ ثانیه (۳ دقیقه) کافیه
                $response = Http::timeout(180)
                    ->connectTimeout(10)  // اگه ۱۰ ثانیه وصل نشد، سریع شکست بخوره
                    ->post("{$pythonServiceUrl}/process", [
                        'user_id' => $this->user->id,
                        'session_id' => $this->session->id,
                        'file_path' => $document->file_path,
                        'document_ids' => [$document->id]
                    ]);
                
                Log::info('DOCUMENT JOB: after Python', [
                    'document_id' => $docId,
                    'status' => $response->status(),
                    'successful' => $response->successful(),
                ]);

                if ($response->successful()) {
                    $document->update([
                        'processing_status' => 'completed',
                        'processed_at' => now(),
                        'processing_error' => null,
                    ]);
                    $successCount++;
                    
                    Log::info("Document {$docId} processed successfully");
                } else {
                    // خطای سرویس پایتون
                    $document->update([
                        'processing_status' => 'failed',
                        'processing_error' => "HTTP {$response->status()}: " . substr($response->body(), 0, 500),
                    ]);
                    
                    Log::error("Failed to process document {$docId}", [
                        'status' => $response->status(),
                        'response' => substr($response->body(), 0, 500)
                    ]);
                    $failedCount++;
                }
                
            } catch (\Illuminate\Http\Client\ConnectionException $e) {
                // timeout یا connection error
                $document->update([
                    'processing_status' => 'failed',
                    'processing_error' => 'Timeout or connection error: ' . $e->getMessage(),
                ]);
                
                Log::error("Timeout/connection error for document {$docId}", [
                    'error' => $e->getMessage()
                ]);
                $failedCount++;
                
            } catch (\Exception $e) {
                $document->update([
                    'processing_status' => 'failed',
                    'processing_error' => $e->getMessage(),
                ]);
                
                Log::error("Exception while processing document {$docId}", [
                    'error' => $e->getMessage()
                ]);
                $failedCount++;
            }
        }

        Log::info('Document processing job completed', [
            'total' => count($this->documentIds),
            'success' => $successCount,
            'failed' => $failedCount
        ]);
    }

    public function failed(\Throwable $exception)
    {
        Log::error('Document processing job failed permanently', [
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds,
            'error' => $exception->getMessage()
        ]);
        
        // همه داکیومنت‌ها رو failed کن
        Document::whereIn('id', $this->documentIds)->update([
            'processing_status' => 'failed',
            'processing_error' => 'Job failed: ' . $exception->getMessage(),
        ]);
    }
}