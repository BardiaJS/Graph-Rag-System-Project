<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class ChatSession extends Model
{
    protected $fillable =[
        'user_id' , '' , 'title'
    ];

    public function user():BelongsTo{
        return $this->belongsTo(User::class , 'user_id');
    }


    public function documents():HasMany{
        return $this->hasMany(Document::class , 'chat_session_id');
    }

    public function questions():HasMany{
        return $this->hasMany(Question::class , 'chat_session_id');
    }
}
