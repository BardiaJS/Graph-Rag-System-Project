<?php

use App\Http\Controllers\Admin\AdminController;
use App\Http\Controllers\Document\DocumentController;
use App\Http\Controllers\Session\SessionController;
use App\Http\Controllers\User\UserController;
use App\Http\Controllers\Webhook\WebhookController;
use Illuminate\Support\Facades\Route;
use Illuminate\Http\Request;


Route::options('/{any}', function () {
    return response()->json([], 200);
})->where('any', '.*');

// ==== Users ====
Route::post('/users/register' , [UserController::class , 'register']);
Route::post('/users/login' , [UserController::class , 'login'])->name('login');
Route::patch('/users/{user}/update' , [UserController::class, 'update'])->middleware('auth:sanctum');
Route::post('/users/logout' , [UserController::class, 'logout'])->middleware('auth:sanctum' ,'not.banned');

Route::get('/user', function (Request $request) {
    return response()->json($request->user());
})->middleware('auth:sanctum');

// ==== Sessions ====
Route::post('/sessions/create/{user}' , [SessionController::class , 'create_session'])->middleware('auth:sanctum' ,'not.banned');
Route::delete('sessions/{session}/delete' , [SessionController::class, 'delete_session'])->middleware('auth:sanctum','not.banned');
Route::patch('sessions/{session}/update' , [SessionController::class, 'update_session'])->middleware('auth:sanctum','not.banned');
Route::get('/sessions/{sessions}'  ,[SessionController::class, 'get_session'])->middleware('auth:sanctum','not.banned');
Route::get('/sessions'  ,[SessionController::class, 'sessions'])->middleware('auth:sanctum','not.banned');

// ==== Documents ====
Route::post('/sessions/{session}/search' , [DocumentController::class, 'search'])->middleware('auth:sanctum','not.banned');
Route::post('webhook/document-processed', [WebhookController::class, 'handle' ]);
Route::post('/documents/upload/sessions/{session}/users/{user}' , [DocumentController::class , 'upload_document'])->middleware('auth:sanctum','not.banned');

Route::delete('/documents/{document}/delete' , [DocumentController::class, 'delete'])->middleware('auth:sanctum','not.banned');
Route::get('/documents/{document}' , [DocumentController::class, 'get_document'])->middleware('auth:sanctum','not.banned');
Route::get('/sessions/{session}/documents' , [DocumentController::class, 'documents'])->middleware('auth:sanctum','not.banned');


// ===== Questions =====
Route::get('/questions/{question}/result', [DocumentController::class, 'getResult'])->middleware('auth:sanctum','not.banned');
Route::get('/questions/{question}/status', [DocumentController::class, 'getStatus'])->middleware('auth:sanctum','not.banned');


// ===== Admins =====
Route::get('/users' , [AdminController::class, 'get_users'])->middleware('auth:sanctum','not.banned');
Route::get('/users/{user}' , [AdminController::class, 'get_user'])->middleware('auth:sanctum','not.banned');
Route::post('/ban/users/{user}' , [AdminController::class , 'ban_user'])->middleware('auth:sanctum','not.banned');
Route::post('/unban/users/{user}' , [AdminController::class , 'unban_user'])->middleware('auth:sanctum','not.banned');