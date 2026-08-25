<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('answers', function (Blueprint $table) {
            $table->id();
            $table->foreignId('question_id')->constrained()->onDelete('cascade');
            $table->text('answer');
            $table->json('sources')->nullable();
            $table->json('sub_queries')->nullable();
            $table->json('search_plans')->nullable();
            $table->integer('num_searches')->default(0);
            $table->integer('version')->default(1);
            $table->boolean('is_latest')->default(true);
            $table->timestamp('answered_at')->nullable();
            $table->timestamps();
            
            // ایندکس برای جستجوی سریع
            $table->index(['question_id', 'is_latest']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('answers');
    }
};
