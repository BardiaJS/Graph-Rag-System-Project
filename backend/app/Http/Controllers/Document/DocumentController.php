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
use App\Services\GraphRagClient;
use App\Services\GraphRagException;
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
     * لیست اسناد یک جلسه (route: GET /sessions/{session}/documents)
     *
     * FIX: this route previously checked only that the user was logged in, so any
     * authenticated user could list the documents of ANY session. It now checks that the
     * session belongs to the user, exactly like getDocuments().
     */
    public function documents(ChatSession $session)
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

        // Cap at 10 documents per question: the agent searches each one, so an
        // unbounded list makes a single question arbitrarily slow.
        $documentIds = Document::where('chat_session_id', $session->id)
            ->where('processing_status', 'completed')
            ->orderBy('id')
            ->limit(10)
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
            ->with('latestAnswer')
            ->first();

        if (!$question) {
            return response()->json([
                'message' => 'Question not found'
            ], 404);
        }

        $answer = $question->latestAnswer;

        // FIX: the job saves the text in `content`; this used to read `answer`.
        return response()->json([
            'question' => $question->content,
            'answer' => $answer?->content ?? $answer?->answer ?? 'پاسخی یافت نشد',
            'status' => $question->status,
            'sources' => $answer->sources ?? null,
            'sub_queries' => $answer->sub_queries ?? null,
            'search_plans' => $answer->search_plans ?? null,
            'warnings' => data_get($answer?->search_plans, 'warnings', []),
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

    // NOTE: the old webhook() method that lived here was a duplicate of
    // WebhookController::handle() and had no route. It was removed so there is a
    // single, authenticated place that handles Python callbacks.

    public function delete(Document $document, GraphRagClient $rag)
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

        // Remove the document's vectors, graph data and figure images from the Python
        // service FIRST. If that fails we keep the database row so the user can retry,
        // instead of orphaning data that nothing could ever delete again.
        try {
            $rag->deleteDocument((int) $document->id, (string) $user->id);
        } catch (GraphRagException $e) {
            Log::error("Python delete failed for document {$document->id}", ['error' => $e->getMessage()]);

            if ($e->getCode() === 409) {
                return response()->json([
                    'message' => 'This document is still being processed. Try again when it has finished.'
                ], 409);
            }

            return response()->json([
                'message' => 'Could not remove the document data right now. Please try again.'
            ], 503);
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
