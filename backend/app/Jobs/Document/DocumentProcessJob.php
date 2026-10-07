<?php

namespace App\Jobs\Document;

use App\Models\ChatSession;
use App\Models\Document;
use App\Models\DocumentChunk;
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

    public $tries = 1;
    public $timeout = 600;  // ← از ۲۴۰ به ۶۰۰ (چون پردازش طول می‌کشه)

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
            'document_ids' => $this->documentIds,
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

                $response = Http::timeout(600)          // ← از ۱۸۰ به ۶۰۰
                    ->connectTimeout(10)
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
                    $data = $response->json('result');
                    $markdown = $data['markdown'] ?? null;
                    $chunks = $data['chunks'] ?? [];

                    // ۱. markdown و وضعیت رو ذخیره کن
                    $document->update([
                        'processing_status' => 'completed',
                        'processed_at' => now(),
                        'processing_error' => null,
                        // 'content' => $markdown,   // ← اگه فیلد content رو اضافه کردی
                        // 'chunks_count' => count($chunks),
                    ]);

                    // ۲. chunkها رو ذخیره کن
                    foreach ($chunks as $idx => $chunk) {
                        DocumentChunk::updateOrCreate(
                            [
                                'document_id' => $document->id,
                                'chunk_index' => $idx,
                            ],
                            [
                                'user_id' => $this->user->id,
                                'session_id' => $this->session->id,
                                'text' => $chunk['text'] ?? '',
                                'enriched_text' => $chunk['enriched_text'] ?? null,
                                'headings' => $chunk['headings'] ?? [],
                                'page' => $chunk['page'] ?? null,
                            ]
                        );
                    }

                    $successCount++;
                    Log::info("Document {$docId} processed successfully", [
                        'chunks_saved' => count($chunks),
                    ]);
                } else {
                    $document->update([
                        'processing_status' => 'failed',
                        'processing_error' => "HTTP {$response->status()}: " . substr($response->body(), 0, 500),
                    ]);

                    Log::error("Failed to process document {$docId}", [
                        'status' => $response->status(),
                        'response' => substr($response->body(), 0, 500),
                    ]);
                    $failedCount++;
                }

            } catch (\Illuminate\Http\Client\ConnectionException $e) {
                $document->update([
                    'processing_status' => 'failed',
                    'processing_error' => 'Timeout or connection error: ' . $e->getMessage(),
                ]);

                Log::error("Timeout/connection error for document {$docId}", [
                    'error' => $e->getMessage(),
                ]);
                $failedCount++;

            } catch (\Exception $e) {
                $document->update([
                    'processing_status' => 'failed',
                    'processing_error' => $e->getMessage(),
                ]);

                Log::error("Exception while processing document {$docId}", [
                    'error' => $e->getMessage(),
                ]);
                $failedCount++;
            }
        }

        Log::info('Document processing job completed', [
            'total' => count($this->documentIds),
            'success' => $successCount,
            'failed' => $failedCount,
        ]);
    }

    public function failed(\Throwable $exception)
    {
        Log::error('Document processing job failed permanently', [
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds,
            'error' => $exception->getMessage(),
        ]);

        Document::whereIn('id', $this->documentIds)->update([
            'processing_status' => 'failed',
            'processing_error' => 'Job failed: ' . $exception->getMessage(),
        ]);
    }
}