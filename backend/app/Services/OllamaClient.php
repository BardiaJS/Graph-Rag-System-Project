<?php

namespace App\Services;

use App\Services\GraphRagException;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

/**
 * Minimal Ollama client. Local only: document text never leaves your infrastructure.
 */
class OllamaClient
{
    /** Strip invalid UTF-8 so json_encode of the request can never fail on odd PDF text. */
    private function clean(string $text): string
    {
        return mb_convert_encoding($text, 'UTF-8', 'UTF-8');
    }

    /**
     * @param  array<int, array{role: string, content: string}>  $messages
     * @param  string|null  $model  overrides the configured model for this call
     * @throws GraphRagException when the model cannot be reached or errors
     */
    public function chat(array $messages, bool $json = false, float $temperature = 0.1, ?string $model = null): string
    {
        $messages = array_map(
            fn ($m) => ['role' => $m['role'], 'content' => $this->clean((string) $m['content'])],
            $messages
        );

        $payload = [
            'model' => $model ?: (string) config('graphrag.ollama.model'),
            'messages' => $messages,
            'stream' => false,
            'keep_alive' => (string) config('graphrag.ollama.keep_alive', '10m'),
            'options' => [
                'temperature' => $temperature,
                'num_ctx' => (int) config('graphrag.ollama.num_ctx', 8192),
                'num_predict' => (int) config('graphrag.ollama.num_predict', 768),
            ],
        ];

        if ($json) {
            $payload['format'] = 'json';
        }

        try {
            $response = Http::baseUrl(rtrim((string) config('graphrag.ollama.url'), '/'))
                ->acceptJson()
                ->connectTimeout(10)
                ->timeout((int) config('graphrag.ollama.timeout', 280))
                ->post('/api/chat', $payload);

            $response->throw();
        } catch (RequestException $e) {
            throw new GraphRagException(
                'Language model error ' . $e->response->status() . ': ' . $e->response->body(),
                $e->response->status(),
                $e
            );
        } catch (ConnectionException $e) {
            throw new GraphRagException('Language model unreachable: ' . $e->getMessage(), 0, $e);
        }

        return (string) data_get($response->json(), 'message.content', '');
    }

    /**
     * Ask for JSON and never throw: on any failure (model down, malformed output)
     * return $default, so planning and grading degrade instead of failing the request.
     *
     * @param  array<int, array{role: string, content: string}>  $messages
     */
    public function chatJson(array $messages, array $default = [], ?string $model = null): array
    {
        try {
            $raw = $this->chat($messages, true, 0.1, $model);
        } catch (GraphRagException $e) {
            Log::warning('Ollama JSON call failed: ' . $e->getMessage());

            return $default;
        }

        $decoded = json_decode($raw, true);
        if (is_array($decoded)) {
            return $decoded;
        }

        // Models sometimes wrap JSON in prose: take the outermost {...}.
        if (preg_match('/\{.*\}/s', $raw, $match)) {
            $decoded = json_decode($match[0], true);
            if (is_array($decoded)) {
                return $decoded;
            }
        }

        Log::warning('Could not parse JSON from the language model: ' . mb_substr($raw, 0, 200));

        return $default;
    }
}