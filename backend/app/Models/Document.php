<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Document extends Model
{
    protected $fillable = [
        'user_id',
        'chat_session_id',
        'name',
        'file_path',
        'page_count',
        'processing_status',
        'processing_error',
        'processed_at',
        'metadata'
    ];

    public function user():BelongsTo{
        return $this->belongsTo(User::class , 'user_id');
    }

    public function session():BelongsTo{
        return $this->belongsTo(ChatSession::class , 'chat_session_id');
    }

    public function chunks(): HasMany
    {
        return $this->hasMany(DocumentChunk::class);
    }
}
