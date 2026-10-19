<?php

namespace App\Http\Controllers\Session;

use App\Http\Controllers\Controller;
use App\Http\Requests\Session\SessionCreateRequest;
use App\Http\Requests\Session\SessionUpdateRequest;
use App\Models\ChatSession;
use App\Models\Question;
use Illuminate\Support\Facades\Auth;

class SessionController extends Controller
{
    /**
     * دریافت لیست همه جلسات کاربر فعلی
     */
    public function sessions()
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $sessions = ChatSession::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json([
            'sessions' => $sessions,
            'total' => $sessions->count()
        ]);
    }

    /**
     * دریافت اطلاعات یک جلسه خاص
     */
    public function get_session(ChatSession $session)
    {
        // ===== ابتدا احراز هویت را بررسی کنید =====
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        // ===== سپس مالکیت جلسه را بررسی کنید =====
        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this session'
            ], 403);
        }

        // ===== دریافت اسناد این جلسه =====
        $documents = $session->documents;

        // ===== دریافت سوالات این جلسه با آخرین پاسخ =====
        $questions = Question::where('chat_session_id', $session->id)
            ->with('latestAnswer')
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json([
            'session' => $session,
            'documents' => $documents,
            'documents_count' => $documents->count(),
            'questions' => $questions,
            'questions_count' => $questions->count()
        ]);
    }

    /**
     * ایجاد جلسه جدید
     */
    public function create_session(SessionCreateRequest $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $validated_data = $request->validated();
        $validated_data['user_id'] = $user->id;
        
        // اگر عنوان وارد نشده، عنوان پیش‌فرض بگذار
        if (empty($validated_data['title'])) {
            $validated_data['title'] = 'جلسه جدید ' . date('Y-m-d H:i');
        }

        $session = ChatSession::create($validated_data);

        return response()->json([
            'message' => 'Session created successfully',
            'session' => $session,
            'id' => $session->id
        ], 201);
    }

    /**
     * به‌روزرسانی جلسه
     */
    public function update_session(ChatSession $session, SessionUpdateRequest $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        // بررسی مالکیت جلسه
        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this session'
            ], 403);
        }

        $validated_data = $request->validated();
        $session->update($validated_data);

        return response()->json([
            'message' => 'Session updated successfully',
            'session' => $session
        ]);
    }

    /**
     * حذف جلسه
     */
    public function delete_session(ChatSession $session)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        // بررسی مالکیت جلسه
        if ($session->user_id !== $user->id) {
            return response()->json([
                'message' => 'You do not have access to this session'
            ], 403);
        }

        // حذف سوالات مرتبط با این جلسه (با توجه به cascade در migration)
        $session->delete();

        return response()->json([
            'message' => 'Session deleted successfully',
            'session_id' => $session->id
        ]);
    }


    public function get_questions_in_the_session(ChatSession $session)
    {
        $user = Auth::user();
        
        if ($session->user_id !== $user->id) {
            return response()->json(['message' => 'Forbidden'], 403);
        }
        
        $questions = $session->questions()
            ->orderBy('created_at', 'asc')
            ->get();
        
        return response()->json([
            'questions' => $questions
        ]);
    }


    public function history(ChatSession $session)
    {
        $user = Auth::user();
        
        if ($session->user_id !== $user->id) {
            return response()->json(['message' => 'Forbidden'], 403);
        }
        
        $questions = $session->questions()
            ->with(['answers' => function ($q) {
                $q->where('is_latest', true);
            }])
            ->orderBy('created_at', 'asc')
            ->get()
            ->map(function ($question) {
                return [
                    'question_id' => $question->id,
                    'question' => $question->content,
                    'status' => $question->status,
                    'created_at' => $question->created_at,
                    'answer' => $question->answers->first() ? [
                        'content' => $question->answers->first()->content,
                        'sources' => $question->answers->first()->sources,
                        'answered_at' => $question->answers->first()->answered_at,
                    ] : null,
                ];
            });
        
        return response()->json([
            'session_id' => $session->id,
            'history' => $questions,
        ]);
    }
}