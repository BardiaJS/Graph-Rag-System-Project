<?php

namespace Tests\Feature\GraphRag;

use Tests\TestCase;

/**
 * The callback route must only accept the Python service. These tests never reach the
 * database: authentication and validation both fail before any query runs.
 */
class CallbackAuthTest extends TestCase
{
    public function test_callback_rejects_a_missing_or_wrong_key(): void
    {
        config(['graphrag.api_key' => 'secret']);

        $this->postJson('/api/graph-rag/callback', [])->assertStatus(401);

        $this->postJson('/api/graph-rag/callback', [], ['X-API-Key' => 'wrong'])->assertStatus(401);
    }

    public function test_callback_with_the_right_key_reaches_validation(): void
    {
        config(['graphrag.api_key' => 'secret']);

        $this->postJson('/api/graph-rag/callback', [], ['X-API-Key' => 'secret'])->assertStatus(422);
    }
}