<?php

namespace App\Http\Controllers\Document;

use App\Http\Controllers\Controller;
use App\Http\Requests\Document\DocumentUploadRequest;
use App\Http\Requests\Question\QuestionCreateRequest;
use App\Jobs\Document\DocumentProcessJob;
use App\Jobs\Rag\AgenticAnswerJob;
use App\Models\ChatSession;
use App\Models\Document;
use App\Models\Question;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;
use Smalot\PdfParser\Parser;

class DocumentController extends Controller
{

    public function upload_document(DocumentUploadRequest $request, ChatSession $session)
    {
        // ۱. دریافت کاربر از احراز هویت
        $user = $request->user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }
        
        // ۲. بررسی مالکیت سشن
        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You are not authorized to upload to this session'
            ], 403);
        }
        
        // ۳. اعتبارسنجی داده‌ها
        $validatedData = $request->validated();
        $validatedData['user_id'] = $user->id;
        $validatedData['chat_session_id'] = $session->id;
        
        // ۴. دریافت فایل‌ها
        $files = $request->file('files');
        
        if (empty($files)) {
            return response()->json([
                'message' => 'No files uploaded'
            ], 400);
        }
        
        $uploadedDocuments = [];
        $documentIds = []; // برای ارسال به Job
        $pdfParser = new Parser();
        
        // ۵. پردازش هر فایل
        foreach ($files as $attachment) {
            // ذخیره فایل
            $path = $attachment->store('documents/' . $session->id, 'public');
            $fullPath = storage_path('app/public/' . $path);
            
            // شمارش صفحات
            try {
                $pdf = $pdfParser->parseFile($fullPath);
                $pageCount = count($pdf->getPages());
            } catch (\Exception $e) {
                $pageCount = 0;
                Log::warning("Failed to parse PDF: " . $e->getMessage());
            }
            
            // ایجاد رکورد در دیتابیس
            $documentData = $validatedData;
            $documentData['file_path'] = $path;
            $documentData['original_name'] = $attachment->getClientOriginalName();
            $documentData['file_size'] = $attachment->getSize();
            $documentData['page_count'] = $pageCount;
            $documentData['mime_type'] = $attachment->getMimeType();

            $document = Document::create($documentData);
            $uploadedDocuments[] = $document;
            $documentIds[] = $document->id; // جمع‌آوری IDها
        }
        
        if (!empty($documentIds)) {
            DocumentProcessJob::dispatch($user, $session, $documentIds);
        }
        
        return response()->json([
            'message' => 'Documents uploaded successfully',
            'documents' => $uploadedDocuments,
            'processing_started' => true
        ], 200);
    }




    public function search(ChatSession $session, QuestionCreateRequest $request)
    {
        $user = Auth::user();
        $validated_data = $request->validated();
        $validated_data['user_id'] = $user->id;
        $validated_data['chat_session_id'] = $session->id;
        $validated_data['status'] = 'pending';

        $question = Question::create($validated_data);

        // دریافت Document های مرتبط با سشن که پردازش کامل شده‌اند
        $documentIds = Document::where('chat_session_id', $session->id)
            ->where('processing_status', 'completed')
            ->pluck('id')
            ->toArray();

        // اگر هیچ سندی وجود ندارد
        if (empty($documentIds)) {
            $question->update([
                'status' => 'failed',
                'error_message' => 'No processed documents found in this session'
            ]);

            return response()->json([
                'message' => 'No processed documents found',
                'question_id' => $question->id
            ], 400);
        }

        // ارسال Job به صف
        AgenticAnswerJob::dispatch($user, $session, $question, $documentIds);

        return response()->json([
            'message' => 'Question processing started',
            'question_id' => $question->id,
            'status' => 'pending',
            'estimated_time' => '30-60 seconds'
        ], 202);
    }

    public function getResult(Request $request, $questionId)
    {
        $user = Auth::user();
        $question = Question::where('id', $questionId)
            ->where('user_id', $user->id)
            ->first();

        if (!$question) {
            return response()->json([
                'message' => 'Question not found'
            ], 404);
        }

        return response()->json([
            'question' => $question->content,
            'answer' => $question->answer,
            'status' => $question->status,
            'sources' => $question->sources,
            'sub_queries' => $question->sub_queries,
            'search_plans' => $question->search_plans,
            'num_searches' => $question->num_searches,
            'created_at' => $question->created_at,
            'answered_at' => $question->answered_at,
            'error' => $question->error_message
        ]);
    }

    public function getStatus($questionId)
    {
        $user = Auth::user();
        $question = Question::where('id', $questionId)
            ->where('user_id', $user->id)
            ->first();

        if (!$question) {
            return response()->json(['message' => 'Question not found'], 404);
        }

        return response()->json([
            'question_id' => $question->id,
            'status' => $question->status,
            'is_completed' => in_array($question->status, ['completed', 'failed'])
        ]);
    }
}
