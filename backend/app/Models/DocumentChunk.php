<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class DocumentChunk extends Model
{
    protected $fillable = [
        'document_id',
        'user_id',
        'session_id',
        'chunk_index',
        'text',
        'enriched_text',
        'headings',
        'page',
        'qdrant_point_id',
    ];

    protected $casts = [
        'headings' => 'array',
        'page' => 'integer',
        'chunk_index' => 'integer',
    ];

    public function document(): BelongsTo
    {
        return $this->belongsTo(Document::class);
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function session(): BelongsTo
    {
        return $this->belongsTo(ChatSession::class);
    }
}