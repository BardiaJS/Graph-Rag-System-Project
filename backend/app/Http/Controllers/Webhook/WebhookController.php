<?php

namespace App\Http\Controllers\Webhook;

use App\Http\Controllers\Controller;
use App\Models\Document;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class WebhookController extends Controller
{
    public function handle(Request $request)
    {
        Log::info('📨 Webhook received', $request->all());
        
        $validated = $request->validate([
            'document_id' => 'required|exists:documents,id',
            'status' => 'required|in:success,failed',
            'pages' => 'nullable|integer',
            'chunks' => 'nullable|integer',
            'entities' => 'nullable|integer',
            'figures' => 'nullable|integer',
            'tables' => 'nullable|integer',
            'qdrant_points' => 'nullable|integer',
            'file_size' => 'nullable|integer'
        ]);

        $document = Document::find($validated['document_id']);

        if ($validated['status'] === 'success') {
            $document->update([
                'processing_status' => 'completed',
                'page_count' => $validated['pages'] ?? $document->page_count,
                'processed_at' => now(),
                'metadata' => array_merge($document->metadata ?? [], [
                    'chunks' => $validated['chunks'] ?? 0,
                    'entities' => $validated['entities'] ?? 0,
                    'figures' => $validated['figures'] ?? 0,
                    'tables' => $validated['tables'] ?? 0,
                    'qdrant_points' => $validated['qdrant_points'] ?? 0,
                ])
            ]);

            Log::info("✅ Document {$document->id} processed successfully via webhook");
        } else {
            $document->update([
                'processing_status' => 'failed',
                'processing_error' => 'Webhook reported failure'
            ]);

            Log::error("❌ Document {$document->id} processing failed via webhook");
        }

        return response()->json(['message' => 'Webhook received'], 200);
    }
}
