<?php

namespace Tests\Feature;

use App\Services\AgentService;
use App\Services\GraphRagClient;
use App\Services\GraphRagException;
use Illuminate\Support\Facades\Http;
use Tests\TestCase;

/**
 * Both the Python service (rag.test) and Ollama (ollama.test) are faked, so these run
 * without a GPU or any running service:   php artisan test --filter=AgentServiceTest
 *
 * Ollama calls happen in this order: plan, [grade], [grade], synthesis.
 * Each Http::sequence() below lists the replies in exactly that order.
 */
class AgentServiceTest extends TestCase
{
    protected function setUp(): void
    {
        parent::setUp();

        config([
            'graphrag.base_url' => 'http://rag.test',
            'graphrag.api_key' => 'secret',
            'graphrag.ollama.url' => 'http://ollama.test',
            'graphrag.ollama.model' => 'test-model',
            'graphrag.agent.max_iterations' => 3,
            'graphrag.agent.max_subquestions' => 4,
            'graphrag.agent.evidence_per_doc' => 6,
        ]);
    }

    private function chat(string $content): array
    {
        return ['message' => ['role' => 'assistant', 'content' => $content]];
    }

    private function evidence(
        int $doc,
        int $idx,
        string $text,
        float $score,
        array $figures = [],
        array $via = []
    ): array {
        return [
            'key' => $doc . ':' . $idx,
            'document_id' => $doc,
            'chunk_index' => $idx,
            'page' => $idx + 1,
            'headings' => ['Section'],
            'text' => $text,
            'score' => $score,
            'linked_via' => $via,
            'figures' => $figures,
        ];
    }

    private function figure(int $doc, int $index, string $number): array
    {
        return [
            'document_id' => $doc,
            'figure_index' => $index,
            'label' => 'Figure ' . $number,
            'number' => $number,
            'caption' => 'Figure ' . $number . ': Overview',
            'page' => 2,
            'relation' => 'DESCRIBES',
            'image_url' => '/documents/' . $doc . '/figures/' . $index . '/image',
        ];
    }

    public function test_full_loop_retrieves_grades_synthesizes_and_verifies(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::sequence()
                ->push(['status' => 'success', 'evidence' => [
                    $this->evidence(7, 1, 'Figure 1: Overview of the architecture.', 0.9, [$this->figure(7, 0, '1')]),
                ]])
                ->push(['status' => 'success', 'evidence' => [
                    $this->evidence(8, 9, 'Paper B also measures F1.', 0.6, [], ['F1']),
                ]]),
            'rag.test/verify' => Http::response([
                'status' => 'success',
                'answer' => 'The architecture is shown in Figure 1 [S1]. Paper B reports F1 [S2].',
                'cited_sids' => ['S1', 'S2'],
                'unverified_figures' => [],
                'warnings' => [],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"architecture","document_ids":[7]}]}'))
                ->push($this->chat('{"sufficient":false,"missing":[{"q":"F1 in paper B","document_ids":[8]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('The architecture is shown in Figure 1 [S1]. Paper B reports F1 [S2].')),
        ]);

        $result = app(AgentService::class)->answer('Compare the papers', [7, 8]);

        $this->assertSame(
            'The architecture is shown in Figure 1 [S1]. Paper B reports F1 [S2].',
            $result['answer']
        );
        $this->assertSame([], $result['warnings']);
        $this->assertCount(2, $result['citations']);

        // Figure travels with its citation; entity link is reported.
        $this->assertSame('S1', $result['citations'][0]['id']);
        $this->assertSame('Figure 1', $result['citations'][0]['figures'][0]['label']);
        $this->assertSame('S2', $result['citations'][1]['id']);
        $this->assertSame(['F1'], $result['citations'][1]['linked_via']);

        // The service is called with the shared secret...
        Http::assertSent(fn ($request) => str_contains($request->url(), '/retrieve')
            && $request->hasHeader('X-API-Key', 'secret'));

        // ...and /verify receives numbered sources with the figure numbers that are linked.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/verify')
            && $request['sources'][0]['sid'] === 'S1'
            && $request['sources'][0]['figure_numbers'] === ['1']
            && $request['sources'][1]['sid'] === 'S2');

        // The second retrieve round used the follow-up query.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/retrieve')
            && ($request['queries'][0]['q'] ?? null) === 'F1 in paper B');
    }

    public function test_the_llm_cannot_widen_the_document_scope(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'Some relevant passage about the topic.', 0.9),
            ]]),
            'rag.test/verify' => Http::response([
                'answer' => 'Answer [S1].', 'cited_sids' => ['S1'], 'unverified_figures' => [], 'warnings' => [],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                // The model "asks" for document 999, which this user does not have.
                ->push($this->chat('{"sub_questions":[{"q":"topic","document_ids":[7,999]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('Answer [S1].')),
        ]);

        app(AgentService::class)->answer('What about the topic?', [7]);

        Http::assertSent(fn ($request) => str_contains($request->url(), '/retrieve')
            && $request['queries'][0]['document_ids'] === [7]
            && $request['document_ids'] === [7]);

        Http::assertNotSent(fn ($request) => str_contains(json_encode($request->data()), '999'));
    }

    public function test_planning_failure_falls_back_to_a_single_query_over_all_documents(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'Relevant text.', 0.8),
            ]]),
            'rag.test/verify' => Http::response([
                'answer' => 'Answer [S1].', 'cited_sids' => ['S1'], 'unverified_figures' => [], 'warnings' => [],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                ->push(['error' => 'model crashed'], 500)      // planning fails
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('Answer [S1].')),
        ]);

        $result = app(AgentService::class)->answer('Original question', [7, 8]);

        $this->assertSame('Answer [S1].', $result['answer']);

        Http::assertSent(fn ($request) => str_contains($request->url(), '/retrieve')
            && $request['queries'][0]['q'] === 'Original question'
            && $request['queries'][0]['document_ids'] === [7, 8]);
    }

    public function test_no_evidence_returns_a_plain_answer_without_calling_synthesis_or_verify(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => []]),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"anything","document_ids":[7]}]}')),
        ]);

        $result = app(AgentService::class)->answer('Unanswerable', [7]);

        $this->assertStringContainsString("couldn't find", $result['answer']);
        $this->assertSame([], $result['citations']);
        $this->assertContains('No evidence retrieved.', $result['warnings']);

        Http::assertNotSent(fn ($request) => str_contains($request->url(), '/verify'));
    }

    public function test_verification_warnings_reach_the_caller(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'Relevant text.', 0.8),
            ]]),
            'rag.test/verify' => Http::response([
                'answer' => 'It scales well as Figure 7 shows [S1].',
                'cited_sids' => ['S1'],
                'unverified_figures' => ['7'],
                'warnings' => ['Answer mentions figure(s) not found in the retrieved sources: Figure 7'],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"scaling","document_ids":[7]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('It scales well as Figure 7 shows [S1][S99].')),
        ]);

        $result = app(AgentService::class)->answer('Does it scale?', [7]);

        $this->assertCount(1, $result['warnings']);
        $this->assertStringContainsString('Figure 7', $result['warnings'][0]);
    }

    public function test_service_errors_surface_as_graph_rag_exception(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['detail' => 'boom'], 500),
        ]);

        $this->expectException(GraphRagException::class);

        app(GraphRagClient::class)->retrieve([['q' => 'x']], [7]);
    }

    public function test_empty_document_list_is_rejected(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        app(AgentService::class)->answer('Question', []);
    }

    public function test_model_history_language_and_context_size_are_applied(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'Relevant text.', 0.8),
            ]]),
            'rag.test/verify' => Http::response([
                'answer' => 'Answer [S1].', 'cited_sids' => ['S1'], 'unverified_figures' => [], 'warnings' => [],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"limitations of X","document_ids":[7]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('Answer [S1].')),
        ]);

        app(AgentService::class)->answer('And its limits?', [7], [
            'model' => 'gemma3:4b',
            'history' => [['question' => 'What is X?', 'answer' => 'X is a method.']],
            'answer_language' => 'Persian (فارسی)',
            'extra_instructions' => 'Do not use Chinese characters.',
        ]);

        // Every model call uses the per-question model...
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && $request['model'] === 'gemma3:4b');

        // ...and asks Ollama for a real context window (the /v1 endpoint ignored this).
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && $request['options']['num_ctx'] === 8192);

        // The planner sees the conversation, so "its limits" can be resolved.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && str_contains($request['messages'][0]['content'], 'What is X?'));

        // The answering rules carry the language requirement.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && $request['messages'][0]['role'] === 'system'
            && str_contains($request['messages'][0]['content'], 'Answer ONLY in Persian')
            && str_contains($request['messages'][0]['content'], 'Chinese'));
    }

    public function test_an_answer_without_citations_is_repaired_once(): void
    {
        $noCitations = [
            'answer' => 'The paper proposes a new method.',
            'cited_sids' => [], 'unverified_figures' => [],
            'warnings' => ['Answer contains no verifiable citations.'],
        ];
        $withCitations = [
            'answer' => 'The paper proposes a new method [S1].',
            'cited_sids' => ['S1'], 'unverified_figures' => [], 'warnings' => [],
        ];

        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'We propose a new method for similarity analysis.', 0.9),
            ]]),
            'rag.test/verify' => Http::sequence()->push($noCitations)->push($withCitations),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"contribution","document_ids":[7]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('The paper proposes a new method.'))          // forgot the citations
                ->push($this->chat('The paper proposes a new method [S1].')),     // repaired
        ]);

        $result = app(AgentService::class)->answer('What is the contribution of this paper?', [7]);

        $this->assertSame('The paper proposes a new method [S1].', $result['answer']);
        $this->assertSame([], $result['warnings']);
        $this->assertCount(1, $result['citations']);
        $this->assertContains('repair', array_column($result['trace'], 'step'));

        // The repair request replays the conversation, then asks for citations.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && count($request['messages']) === 4
            && $request['messages'][2]['role'] === 'assistant'
            && str_contains($request['messages'][3]['content'], 'no source citations'));
    }

    public function test_when_repair_also_fails_the_original_warning_is_kept(): void
    {
        $noCitations = [
            'answer' => 'No markers here.',
            'cited_sids' => [], 'unverified_figures' => [],
            'warnings' => ['Answer contains no verifiable citations.'],
        ];

        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'Some passage.', 0.9),
            ]]),
            'rag.test/verify' => Http::response($noCitations),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"x","document_ids":[7]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('No markers here.'))
                ->push($this->chat('Still no markers.')),
        ]);

        $result = app(AgentService::class)->answer('Question?', [7]);

        $this->assertSame('No markers here.', $result['answer']);
        $this->assertSame([], $result['citations']);
        $this->assertContains('Answer contains no verifiable citations.', $result['warnings']);
    }

    public function test_the_prompt_explains_that_this_paper_means_the_sources(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::response(['evidence' => [
                $this->evidence(7, 1, 'Some passage.', 0.9),
            ]]),
            'rag.test/verify' => Http::response([
                'answer' => 'A [S1].', 'cited_sids' => ['S1'], 'unverified_figures' => [], 'warnings' => [],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"x","document_ids":[7]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('A [S1].')),
        ]);

        app(AgentService::class)->answer('What is the main contribution of this paper?', [7]);

        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && ($request['messages'][0]['role'] ?? '') === 'system'
            && str_contains($request['messages'][0]['content'], 'never claim that no paper'));
    }

    public function test_planner_and_grader_prompts_keep_the_loop_on_topic(): void
    {
        Http::fake([
            'rag.test/retrieve' => Http::sequence()
                ->push(['evidence' => [$this->evidence(7, 1, 'First passage.', 0.9)]])
                ->push(['evidence' => [$this->evidence(8, 2, 'Second passage.', 0.8)]]),
            'rag.test/verify' => Http::response([
                'answer' => 'A [S1].', 'cited_sids' => ['S1'], 'unverified_figures' => [], 'warnings' => [],
            ]),
            'ollama.test/api/chat' => Http::sequence()
                ->push($this->chat('{"sub_questions":[{"q":"topic","document_ids":[7,8]}]}'))
                ->push($this->chat('{"sufficient":false,"missing":[{"q":"topic detail","document_ids":[7,8]}]}'))
                ->push($this->chat('{"sufficient":true}'))
                ->push($this->chat('A [S1].')),
        ]);

        app(AgentService::class)->answer('Tell me about the topic', [7, 8]);

        // The planner is told it only knows ids, and must not silently drop documents.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && str_contains($request['messages'][0]['content'], 'ALL available document ids'));

        // The grader is told to stay on the original question.
        Http::assertSent(fn ($request) => str_contains($request->url(), '/api/chat')
            && str_contains($request['messages'][0]['content'], 'about the ORIGINAL question'));
    }
}