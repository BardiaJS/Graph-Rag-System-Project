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
    /**
     * دریافت لیست اسناد یک جلسه خاص
     */
    public function getDocuments(ChatSession $session)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this session'
            ], 403);
        }

        $documents = Document::where('chat_session_id', $session->id)
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json([
            'documents' => $documents,
            'total' => $documents->count()
        ]);
    }

    /**
     * دریافت یک سند خاص
     */
    public function get_document(Document $document)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        // بررسی مالکیت سند (از طریق Session)
        $session = ChatSession::find($document->chat_session_id);
        if (!$session || $session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this document'
            ], 403);
        }

        return response()->json([
            'document' => $document
        ]);
    }

    /**
     * دریافت همه اسناد کاربر (بدون فیلتر Session)
     */
    public function documents(ChatSession $session)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $documents = Document::where('chat_session_id', $session->id)
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json([
            'documents' => $documents,
            'total' => $documents->count()
        ]);
    }

    /**
     * دریافت سوالات یک جلسه
     */
    public function getQuestions(ChatSession $session)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this session'
            ], 403);
        }

        $questions = Question::where('chat_session_id', $session->id)
            ->with('latestAnswer')
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json([
            'questions' => $questions,
            'total' => $questions->count()
        ]);
    }

    /**
     * آپلود اسناد
     */
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
        $documentIds = [];
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
            $documentIds[] = $document->id;
        }
        
        // ۶. ارسال Job برای پردازش
        if (!empty($documentIds)) {
            DocumentProcessJob::dispatch($user, $session, $documentIds);
        }
        
        return response()->json([
            'message' => 'Documents uploaded successfully',
            'documents' => $uploadedDocuments,
            'document_ids' => $documentIds,
            'processing_started' => true
        ], 200);
    }

    /**
     * پرسش سوال
     */
    public function search(ChatSession $session, QuestionCreateRequest $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this session'
            ], 403);
        }

        $validated_data = $request->validated();
        $validated_data['user_id'] = $user->id;
        $validated_data['chat_session_id'] = $session->id;
        $validated_data['status'] = 'pending';

        $question = Question::create($validated_data);

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

    /**
     * دریافت نتیجه سوال
     */
    public function getResult(Request $request, $questionId)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $question = Question::where('id', $questionId)
            ->where('user_id', $user->id)
            ->with('latestAnswer')  // ← این مهم است
            ->first();

        if (!$question) {
            return response()->json([
                'message' => 'Question not found'
            ], 404);
        }

        $answer = $question->latestAnswer;  // ← دریافت از جدول answers

        return response()->json([
            'question' => $question->content,
            'answer' => $answer->answer ?? 'پاسخی یافت نشد',  // ← از answers
            'status' => $question->status,
            'sources' => $answer->sources ?? null,
            'sub_queries' => $answer->sub_queries ?? null,
            'search_plans' => $answer->search_plans ?? null,
            'num_searches' => $answer->num_searches ?? 0,
            'created_at' => $question->created_at,
            'answered_at' => $answer->answered_at ?? null,
            'error' => $question->error_message
        ]);
    }

    /**
     * دریافت وضعیت سوال
     */
    public function getStatus($questionId)
    {   
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

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

    /**
     * Webhook برای دریافت نتیجه پردازش از Python
     */
    public function webhook(Request $request)
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


    public function delete(Document $document)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $session = ChatSession::find($document->chat_session_id);
        
        if (!$session || $session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have permission to delete this document'
            ], 403);
        }

        $filePath = storage_path('app/public/' . $document->file_path);
        if (file_exists($filePath)) {
            unlink($filePath);
        }

        $document->delete();

        return response()->json([
            'message' => 'Document deleted successfully',
            'document_id' => $document->id
        ], 200);
    }
}