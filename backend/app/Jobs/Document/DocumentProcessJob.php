<?php

namespace App\Jobs\Document;

use App\Models\ChatSession;
use App\Models\Document;
use App\Models\User;
use App\Services\GraphRagClient;
use App\Services\GraphRagException;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Foundation\Queue\Queueable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;

/**
 * Hands each uploaded PDF to the Python service and returns immediately.
 *
 * Python now processes documents in its own background queue, so /process answers
 * "202 queued" in milliseconds. This job therefore NEVER marks a document `completed`:
 * it only sets `processing`. The final state (completed / failed) arrives later through
 * POST /api/graph-rag/callback (see WebhookController).
 */
class DocumentProcessJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 1;
    public $timeout = 120; // only enqueues now; the heavy work happens in Python

    protected $user;
    protected $session;
    protected $documentIds;

    public function __construct(User $user, ChatSession $session, array $documentIds)
    {
        $this->user = $user;
        $this->session = $session;
        $this->documentIds = $documentIds;
    }

    public function handle(GraphRagClient $rag)
    {
        Log::info('Starting document processing job', [
            'user_id' => $this->user->id,
            'session_id' => $this->session->id,
            'document_ids' => $this->documentIds,
        ]);

        $queued = 0;
        $failed = 0;

        foreach ($this->documentIds as $docId) {
            $document = Document::find($docId);

            if (!$document) {
                Log::warning("Document {$docId} not found");
                $failed++;
                continue;
            }

            try {
                $response = $rag->processDocument(
                    (int) $this->user->id,
                    (int) $this->session->id,
                    (string) $document->file_path,
                    [(int) $document->id]
                );

                $document->update([
                    'processing_status' => 'processing',
                    'processing_error' => null,
                    'metadata' => array_merge($document->metadata ?? [], [
                        'job_id' => $response['job_id'] ?? null,
                    ]),
                ]);

                $queued++;
                Log::info("Document {$docId} queued in Python", [
                    'job_id' => $response['job_id'] ?? null,
                    'deduplicated' => $response['deduplicated'] ?? false,
                ]);
            } catch (GraphRagException $e) {
                $document->update([
                    'processing_status' => 'failed',
                    'processing_error' => substr($e->getMessage(), 0, 500),
                ]);

                Log::error("Could not queue document {$docId}", ['error' => $e->getMessage()]);
                $failed++;
            } catch (\Exception $e) {
                $document->update([
                    'processing_status' => 'failed',
                    'processing_error' => substr($e->getMessage(), 0, 500),
                ]);

                Log::error("Exception while queueing document {$docId}", ['error' => $e->getMessage()]);
                $failed++;
            }
        }

        Log::info('Document processing job completed', [
            'total' => count($this->documentIds),
            'queued' => $queued,
            'failed' => $failed,
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
