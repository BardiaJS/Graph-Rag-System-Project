<?php

namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Http\Requests\User\UserLoginRequest;
use App\Http\Requests\User\UserRegisterRequest;
use App\Http\Requests\User\UserUpdateRequest;
use App\Models\User;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;

class UserController extends Controller
{
    // register
    public function register(UserRegisterRequest $userRegisterRequest){
        $validated_data = $userRegisterRequest->validated();
        $user = User::create($validated_data);
        return response()->json([
            'user' => $user
        ]);

    }

    // login
    public function login(UserLoginRequest $userLoginRequest){
        $validated_data = $userLoginRequest->validated();
        if(Auth::attempt($validated_data)){
            $user = Auth::user();
            $token = $user->createToken('acces_token');
            return response()->json([
                'user' => $user ,
                'token' => $token
            ]);
        }
    }
    // update
    public function update(User $user , UserUpdateRequest $userUpdateRequest){
        $validated_data = $userUpdateRequest->validated();
        if(Hash::check($validated_data['password'] , $user->password)){
            $user->update($validated_data);
        }
    }

    public function logout(){
        Auth::logout();
    }
}
