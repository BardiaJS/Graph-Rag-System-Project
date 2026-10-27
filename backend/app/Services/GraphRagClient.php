<?php

namespace App\Services;

use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\PendingRequest;
use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\Http;

/**
 * Thin wrapper around the Python document-processing service.
 * Every method returns decoded JSON (array) or throws GraphRagException.
 */
class GraphRagClient
{
    private function http(int $timeout): PendingRequest
    {
        $request = Http::baseUrl(rtrim((string) config('graphrag.base_url'), '/'))
            ->acceptJson()
            ->timeout($timeout);

        $key = config('graphrag.api_key');
        if ($key) {
            $request = $request->withHeaders(['X-API-Key' => (string) $key]);
        }

        return $request;
    }

    /** Append a query string, skipping null values. */
    private function withQuery(string $path, array $query): string
    {
        $clean = array_filter($query, fn ($value) => $value !== null);

        return $clean ? $path . '?' . http_build_query($clean) : $path;
    }

    private function request(string $method, string $path, array $payload = [], ?int $timeout = null): array
    {
        $http = $this->http($timeout ?? (int) config('graphrag.timeout', 60));

        try {
            if ($method === 'POST') {
                $response = $http->post($path, $payload);
            } elseif ($method === 'DELETE') {
                $response = $http->delete($this->withQuery($path, $payload));
            } else {
                $response = $http->get($this->withQuery($path, $payload));
            }

            $response->throw();
        } catch (RequestException $e) {
            throw new GraphRagException(
                'Graph RAG service error ' . $e->response->status() . ': ' . $e->response->body(),
                $e->response->status(),
                $e
            );
        } catch (ConnectionException $e) {
            throw new GraphRagException('Graph RAG service unreachable: ' . $e->getMessage(), 0, $e);
        }

        return $response->json() ?? [];
    }

    // ------------------------------------------------------------ ingestion

    /** Enqueue a PDF for processing. Returns ['job_id' => 'doc-12', 'document_id' => 12, ...]. */
    public function processDocument(int $userId, int $sessionId, string $filePath, array $documentIds): array
    {
        return $this->request('POST', '/process', [
            'user_id' => $userId,
            'session_id' => $sessionId,
            'file_path' => $filePath,
            'document_ids' => array_values($documentIds),
        ]);
    }

    public function jobStatus(string $jobId): array
    {
        return $this->request('GET', '/jobs/' . rawurlencode($jobId));
    }

    // ------------------------------------------------------------ retrieval

    /**
     * @param  array<int, array{q: string, document_ids?: array<int>}>  $queries
     * @param  array<int>  $documentIds  the caller's allowed documents (hard scope limit)
     * @return array<int, array<string, mixed>>  evidence items, each with a stable 'key'
     */
    public function retrieve(array $queries, array $documentIds): array
    {
        $data = $this->request('POST', '/retrieve', [
            'queries' => array_values($queries),
            'document_ids' => array_values($documentIds),
        ]);

        return $data['evidence'] ?? [];
    }

    /**
     * @param  array<int, array{sid: string, text: string, figure_numbers: array<string>}>  $sources
     * @return array{answer: string, cited_sids: array<string>, unverified_figures: array<string>, warnings: array<string>}
     */
    public function verify(string $answer, array $sources): array
    {
        $data = $this->request('POST', '/verify', [
            'answer' => $answer,
            'sources' => array_values($sources),
        ]);

        return [
            'answer' => (string) ($data['answer'] ?? $answer),
            'cited_sids' => $data['cited_sids'] ?? [],
            'unverified_figures' => $data['unverified_figures'] ?? [],
            'warnings' => $data['warnings'] ?? [],
        ];
    }

    // ------------------------------------------------------------ documents

    public function listDocuments(?string $userId = null, int $limit = 50, int $offset = 0): array
    {
        return $this->request('GET', '/documents', [
            'user_id' => $userId,
            'limit' => $limit,
            'offset' => $offset,
        ])['documents'] ?? [];
    }

    public function getDocument(int $documentId, ?string $userId = null): array
    {
        return $this->request('GET', '/documents/' . $documentId, ['user_id' => $userId])['document'] ?? [];
    }

    public function deleteDocument(int $documentId, ?string $userId = null): array
    {
        return $this->request('DELETE', '/documents/' . $documentId, ['user_id' => $userId]);
    }

    public function listFigures(int $documentId): array
    {
        return $this->request('GET', '/documents/' . $documentId . '/figures')['figures'] ?? [];
    }

    /** Raw PNG bytes of a figure, for a controller that streams it after checking ownership. */
    public function figureImage(int $documentId, int $figureIndex): string
    {
        $http = $this->http((int) config('graphrag.timeout', 60));

        try {
            $response = $http->get('/documents/' . $documentId . '/figures/' . $figureIndex . '/image');
            $response->throw();
        } catch (RequestException $e) {
            throw new GraphRagException('Figure not available: ' . $e->response->status(), $e->response->status(), $e);
        } catch (ConnectionException $e) {
            throw new GraphRagException('Graph RAG service unreachable: ' . $e->getMessage(), 0, $e);
        }

        return $response->body();
    }
}
