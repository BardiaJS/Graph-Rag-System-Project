<?php

namespace App\Http\Controllers\Session;

use App\Http\Controllers\Controller;
use App\Http\Requests\Session\SessionCreateRequest;
use App\Http\Requests\Session\SessionUpdateRequest;
use App\Models\ChatSession;
use App\Models\User;
use Illuminate\Support\Facades\Auth;

class SessionController extends Controller
{
    // create session
    public function create_session(SessionCreateRequest $sessionCreateRequest , User $user){
        if(Auth::user()->id == $user -> id){
            $validated_data = $sessionCreateRequest->validated();
            $validated_data['user_id'] = $user->id; 
            $session = ChatSession::create($validated_data);
            return response()->json([
                'session' => $session
            ]);
        }
    }

    // update session
    public function update_session(ChatSession $session,  SessionUpdateRequest $sessionUpdateRequest){
        $validated_data = $sessionUpdateRequest->validated();
        $session->update($validated_data);
        return response()->json([
            'session' => $session
        ]);

    }

    // delete session
    public function delete_session(ChatSession $session){
        $session->delete();
    }
}
