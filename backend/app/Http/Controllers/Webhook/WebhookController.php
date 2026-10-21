<?php

namespace App\Http\Controllers\Webhook;

use App\Http\Controllers\Controller;
use App\Models\Document;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

/**
 * Receives the final result of a document job from the Python service.
 * Python sends the shared secret in the X-API-Key header.
 */
class WebhookController extends Controller
{
    public function handle(Request $request)
    {
        $this->authorizeCallback($request);

        Log::info('Graph RAG callback received', $request->all());

        $validated = $request->validate([
            'document_id' => 'required|exists:documents,id',
            'status' => 'required|in:success,failed',
            'job_id' => 'nullable|string',
            'pages' => 'nullable|integer',
            'chunks' => 'nullable|integer',
            'entities' => 'nullable|integer',
            'figures' => 'nullable|integer',
            'tables' => 'nullable|integer',
            'qdrant_points' => 'nullable|integer',
            'file_size' => 'nullable|integer',
            'ocr_used' => 'nullable|boolean',
            'pdf_kind' => 'nullable|string',
            'entity_error' => 'nullable|string',
            'figure_error' => 'nullable|string',
            'warning' => 'nullable|string',
            'error' => 'nullable|string',
        ]);

        $document = Document::find($validated['document_id']);

        // "Succeeded" with nothing indexed is useless for answering questions
        // (e.g. a scanned PDF with no OCR): report it as a failure the user can understand.
        $chunks = $validated['chunks'] ?? null;
        $isEmpty = $validated['status'] === 'success' && $chunks === 0;

        if ($validated['status'] === 'success' && !$isEmpty) {
            $document->update([
                'processing_status' => 'completed',
                'processing_error' => null,
                'page_count' => $validated['pages'] ?? $document->page_count,
                'processed_at' => now(),
                'metadata' => array_merge($document->metadata ?? [], [
                    'job_id' => $validated['job_id'] ?? ($document->metadata['job_id'] ?? null),
                    'chunks' => $chunks ?? 0,
                    'entities' => $validated['entities'] ?? 0,
                    'figures' => $validated['figures'] ?? 0,
                    'tables' => $validated['tables'] ?? 0,
                    'qdrant_points' => $validated['qdrant_points'] ?? 0,
                    'ocr_used' => $validated['ocr_used'] ?? false,
                    'pdf_kind' => $validated['pdf_kind'] ?? null,
                    'entity_error' => $validated['entity_error'] ?? null,
                    'figure_error' => $validated['figure_error'] ?? null,
                ]),
            ]);

            // chunks_count is not mass-assignable on the model, so set it explicitly.
            $document->forceFill(['chunks_count' => $chunks ?? 0])->save();

            Log::info("Document {$document->id} processed successfully via callback");
        } else {
            $reason = $validated['error']
                ?? $validated['warning']
                ?? ($isEmpty ? 'No text could be extracted from this document' : 'Processing failed');

            $document->update([
                'processing_status' => 'failed',
                'processing_error' => substr($reason, 0, 1000),
            ]);

            Log::error("Document {$document->id} processing failed via callback", ['reason' => $reason]);
        }

        return response()->json(['message' => 'Callback received'], 200);
    }

    /**
     * Only the Python service may call this route. Without this check anyone could mark
     * any document completed or failed. In local/testing, a missing key is tolerated.
     */
    private function authorizeCallback(Request $request): void
    {
        $expected = (string) config('graphrag.api_key');

        if ($expected === '') {
            abort_unless(app()->environment(['local', 'testing']), 503, 'Callback key not configured');

            return;
        }

        if (!hash_equals($expected, (string) $request->header('X-API-Key'))) {
            abort(401, 'Invalid API key');
        }
    }
}