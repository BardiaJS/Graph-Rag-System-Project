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
        Schema::create('documents', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('chat_session_id')->constrained()->cascadeOnDelete();
            $table->string('name')->nullable();
            $table->string('file_path');
            $table->unsignedInteger('page_count');
            $table->enum('processing_status', ['pending', 'processing', 'completed', 'failed'])
                ->default('pending')
                ->after('page_count');
            
            // زمان اتمام پردازش
            $table->timestamp('processed_at')->nullable()->after('processing_status');
            
            // خطای پردازش (در صورت وجود)
            $table->text('processing_error')->nullable()->after('processed_at');
            
            // متادیتای اضافی (چانک‌ها، موجودیت‌ها و ...)
            $table->json('metadata')->nullable()->after('processing_error');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('documents');
    }
};
