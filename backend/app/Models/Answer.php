<?php

namespace App\Models;  // ← این باید درست باشد

use App\Models\Question;  // ← این را اضافه کنید
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class Answer extends Model
{
    protected $fillable = [
        'question_id',
        'answer',
        'sources',
        'sub_queries',
        'search_plans',
        'num_searches',
        'version',
        'is_latest',
        'answered_at'
    ];

    protected $casts = [
        'sources' => 'array',
        'sub_queries' => 'array',
        'search_plans' => 'array',
        'answered_at' => 'datetime'
    ];

    // رابطه با Question
    public function question(): BelongsTo
    {
        return $this->belongsTo(Question::class);
    }
}