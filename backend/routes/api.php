<?php

use App\Http\Controllers\Document\DocumentController;
use App\Http\Controllers\Session\SessionController;
use App\Http\Controllers\User\UserController;
use App\Http\Controllers\Webhook\WebhookController;
use Illuminate\Support\Facades\Route;



Route::options('/{any}', function () {
    return response()->json([], 200);
})->where('any', '.*');

// user related
Route::post('/users/register' , [UserController::class , 'register']);
Route::post('/users/login' , [UserController::class , 'login'])->name('login');
Route::patch('/users/{user}/update' , [UserController::class, 'update'])->middleware('auth:sanctum');
Route::post('/users/logout' , [UserController::class, 'logout'])->middleware('auth:sanctum');

// session related
Route::post('/sessions/create/{user}' , [SessionController::class , 'create_session'])->middleware('auth:sanctum');
Route::delete('sessions/{session}/delete' , [SessionController::class, 'delete_session'])->middleware('auth:sanctum');
Route::patch('sessions/{session}/update' , [SessionController::class, 'update_session'])->middleware('auth:sanctum');
Route::get('/sessions/{sessions}'  ,[SessionController::class, 'get_session'])->middleware('auth:sanctum');
Route::get('/sessions'  ,[SessionController::class, 'sessions'])->middleware('auth:sanctum');

// document related
Route::post('/sessions/{session}/search' , [DocumentController::class, 'search'])->middleware('auth:sanctum');
Route::post('webhook/document-processed', [WebhookController::class, 'handle']);
Route::post('/documents/upload/sessions/{session}/users/{user}' , [DocumentController::class , 'upload_document'])->middleware('auth:sanctum');
Route::get('/documents/{document}' , [DocumentController::class, 'get_document'])->middleware('auth:sanctum');
Route::get('/documents' , [DocumentController::class, 'documents'])->middleware('auth:sanctum');