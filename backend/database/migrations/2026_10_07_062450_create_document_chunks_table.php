<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('document_chunks', function (Blueprint $table) {
            $table->id();
            
            $table->foreignId('document_id')
                  ->constrained()
                  ->onDelete('cascade');
            
            $table->foreignId('user_id')
                  ->nullable()
                  ->constrained()
                  ->onDelete('cascade');
            
            // ← اینجا مهمه: constrained('chat_sessions')
            $table->foreignId('session_id')
                  ->nullable()
                  ->constrained('chat_sessions')
                  ->onDelete('set null');
            
            $table->integer('chunk_index')->nullable();
            $table->text('text');
            $table->text('enriched_text')->nullable();
            $table->json('headings')->nullable();
            $table->integer('page')->nullable();
            $table->string('qdrant_point_id')->nullable()->index();
            
            $table->timestamps();
            
            $table->index(['document_id', 'chunk_index']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('document_chunks');
    }
};