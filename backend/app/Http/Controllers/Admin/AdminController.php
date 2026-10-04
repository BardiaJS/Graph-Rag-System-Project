<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;

class AdminController extends Controller
{
    public function ban_user(User $user)
    {
        // بررسی اینکه کاربر خودش رو بن نکنه
        if (Auth::user()->id == $user->id) {
            return response()->json([
                'message' => 'You cannot ban yourself!'
            ], 403);
        }

        // بن کردن کاربر
        $user->is_ban = true;
        $user->save();

        Log::info("User {$user->id} banned by admin " . Auth::user()->id);

        return response()->json([
            'message' => 'User banned successfully',
            'user' => $user
        ]);
    }

    public function unban_user(User $user)
    {
        // بررسی اینکه کاربر خودش رو آنبن نکنه
        if (Auth::user()->id == $user->id) {
            return response()->json([
                'message' => 'You cannot unban yourself!'
            ], 403);
        }

        // آنبن کردن کاربر
        $user->is_ban = false;
        $user->save();

        Log::info("User {$user->id} unbanned by admin " . Auth::user()->id);

        return response()->json([
            'message' => 'User unbanned successfully',
            'user' => $user
        ]);
    }

    public function get_users()
    {
        $users = User::paginate(5);
        return response()->json([
            'users' => $users
        ]);
    }

    public function get_user(User $user)
    {
        return response()->json([
            'user' => $user
        ]);
    }
}