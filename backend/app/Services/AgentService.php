<?php

namespace App\Services;

use App\Services\GraphRagClient;
use InvalidArgumentException;

/**
 * Agentic question answering over one or more documents.
 *
 *   plan       split the question into sub-queries (each targeting documents)
 *   retrieve   POST /retrieve on the Python service (vector + neighbours + entity hop + figures)
 *   grade      LLM decides whether the evidence is enough, else proposes follow-up queries
 *   (repeat retrieve/grade up to max_iterations)
 *   synthesize answer ONLY from numbered sources, citing [S1], [S2] ...
 *   verify     POST /verify: strips citations to unknown sources, flags invented figures
 *
 * SECURITY: the caller must already have checked that the user owns every id in
 * $documentIds. The Python service enforces that nothing outside this list is searched,
 * and the LLM can never widen it (see parseQueries()).
 *
 * Options accepted by answer():
 *   model              string  Ollama model for this question (default: config)
 *   history            array   [['question' => '...', 'answer' => '...'], ...] oldest first
 *   answer_language    string  e.g. 'Persian (فارسی)'; default: language of the question
 *   extra_instructions string  appended to the answering rules (e.g. "Do not use Chinese characters.")
 */
class AgentService
{
    private const PLAN_PROMPT = <<<'TXT'
You plan retrieval for a research assistant.
Available document ids: {ids}

Break the user's question into at most {max} short, self-contained
search queries. If the question compares or spans documents, create one query per
relevant document. If it is simple, return a single query. Write queries so they make
sense on their own: if the question refers to the earlier conversation ("it", "the
second one"), replace the reference with what it means.

Return JSON only:
{"sub_questions": [{"q": "<query>", "document_ids": [<ids from the list>]}]}

{history}Question: {question}
TXT;

    private const GRADE_PROMPT = <<<'TXT'
You judge whether retrieved passages are enough to answer a question.

Question: {question}
Available document ids: {ids}

Passages:
{passages}

If the passages contain what is needed (for every document the question concerns),
answer sufficient=true. Otherwise list up to 3 NEW, differently-worded search queries
that target what is missing.

Return JSON only:
{"sufficient": true, "missing": [{"q": "<query>", "document_ids": [<ids>]}]}
TXT;

    private const SYNTH_RULES = 'You answer research questions strictly from the provided sources. '
        . 'Rules: (1) Use only facts stated in the sources. (2) After every claim, cite the '
        . 'supporting source ids like [S2] or [S1][S4]. (3) When documents differ, say which '
        . 'document each statement comes from. (4) Mention a figure only if it is listed under '
        . '"Linked figures" of a source or named in that source\'s text, and cite that source; '
        . 'never invent figure numbers, tables or pages. (5) If the sources do not contain the '
        . 'answer, say so plainly instead of guessing. ';

    public function __construct(
        private GraphRagClient $rag,
        private OllamaClient $llm,
    ) {
    }

    /**
     * @param  array<int>  $documentIds  documents the user owns and wants to ask about
     * @param  array<string, mixed>  $options  see class docblock
     * @return array{answer: string, citations: array, warnings: array, trace: array}
     */
    public function answer(string $question, array $documentIds, array $options = []): array
    {
        $documentIds = array_values(array_unique(array_map('intval', $documentIds)));
        if (count($documentIds) === 0) {
            throw new InvalidArgumentException('documentIds must not be empty');
        }

        $maxIterations = max(1, (int) config('graphrag.agent.max_iterations', 3));
        $trace = [];
        $store = [];

        $pending = $this->plan($question, $documentIds, $options, $trace);

        for ($i = 1; $i <= $maxIterations; $i++) {
            $trace[] = ['step' => 'iteration', 'n' => $i];
            $new = $this->retrieveInto($pending, $documentIds, $store, $trace);

            // Stop when a follow-up round found nothing new, or on the last round.
            if ($i > 1 && $new === 0) {
                $trace[] = ['step' => 'stop', 'reason' => 'no new evidence'];
                break;
            }
            if ($i === $maxIterations) {
                break;
            }

            $pending = $this->grade($question, $documentIds, $store, $options, $trace);
            if (count($pending) === 0) {
                break;
            }
        }

        $evidence = $this->select($store);
        if (count($evidence) === 0) {
            return [
                'answer' => "I couldn't find any relevant content in the selected documents.",
                'citations' => [],
                'warnings' => ['No evidence retrieved.'],
                'trace' => $trace,
            ];
        }

        // Source ids are assigned here, in reading order.
        foreach (array_keys($evidence) as $index) {
            $evidence[$index]['sid'] = 'S' . ($index + 1);
        }

        $raw = $this->synthesize($question, $evidence, $options);
        $check = $this->rag->verify($raw, $this->toSources($evidence));

        $cited = array_flip($check['cited_sids']);
        $citations = [];
        foreach ($evidence as $e) {
            if (isset($cited[$e['sid']])) {
                $citations[] = [
                    'id' => $e['sid'],
                    'document_id' => $e['document_id'],
                    'page' => $e['page'] ?? null,
                    'headings' => $e['headings'] ?? [],
                    'chunk_index' => $e['chunk_index'] ?? null,
                    'quote' => mb_substr((string) ($e['text'] ?? ''), 0, 300),
                    'score' => $e['score'] ?? null,
                    'figures' => $e['figures'] ?? [],
                    'linked_via' => $e['linked_via'] ?? [],
                ];
            }
        }

        $trace[] = [
            'step' => 'synthesize',
            'sources_in_context' => count($evidence),
            'sources_cited' => count($citations),
        ];

        return [
            'answer' => $check['answer'],
            'citations' => $citations,
            'warnings' => $check['warnings'],
            'trace' => $trace,
        ];
    }

    // -------------------------------------------------------------- history

    private function historyText(array $options): string
    {
        $history = $options['history'] ?? [];
        if (!is_array($history) || count($history) === 0) {
            return '';
        }

        $limit = (int) config('graphrag.agent.history_answer_chars', 300);
        $text = "Previous conversation:\n";
        foreach ($history as $turn) {
            $text .= 'User: ' . ($turn['question'] ?? '') . "\n";
            if (!empty($turn['answer'])) {
                $text .= 'Assistant: ' . mb_substr((string) $turn['answer'], 0, $limit) . "\n";
            }
            $text .= "\n";
        }

        return $text . "---\n\n";
    }

    private function model(array $options): ?string
    {
        $model = $options['model'] ?? null;

        return is_string($model) && $model !== '' ? $model : null;
    }

    // ------------------------------------------------------------- planning

    /** @return array<int, array{q: string, document_ids: array<int>}> */
    private function plan(string $question, array $docIds, array $options, array &$trace): array
    {
        $max = max(1, (int) config('graphrag.agent.max_subquestions', 4));

        $prompt = strtr(self::PLAN_PROMPT, [
            '{ids}' => json_encode($docIds),
            '{max}' => (string) $max,
            '{history}' => $this->historyText($options),
            '{question}' => $question,
        ]);

        $data = $this->llm->chatJson(
            [['role' => 'user', 'content' => $prompt]],
            [],
            $this->model($options)
        );
        $subs = $this->parseQueries($data['sub_questions'] ?? null, $docIds);

        if (count($subs) === 0) {
            // Fallback: one pass over every allowed document.
            $subs = [['q' => $question, 'document_ids' => $docIds]];
        }

        $subs = array_slice($subs, 0, $max);
        $trace[] = ['step' => 'plan', 'sub_questions' => $subs];

        return $subs;
    }

    /**
     * Validate LLM output. The model can never widen scope beyond $allowed.
     *
     * @return array<int, array{q: string, document_ids: array<int>}>
     */
    private function parseQueries(mixed $raw, array $allowed): array
    {
        $out = [];
        if (!is_array($raw)) {
            return $out;
        }

        foreach ($raw as $item) {
            if (!is_array($item)) {
                continue;
            }

            $q = isset($item['q']) && is_string($item['q']) ? trim($item['q']) : '';
            if ($q === '') {
                continue;
            }

            $ids = [];
            foreach ((array) ($item['document_ids'] ?? []) as $d) {
                if (!is_numeric($d)) {
                    continue;
                }
                $d = (int) $d;
                if (in_array($d, $allowed, true) && !in_array($d, $ids, true)) {
                    $ids[] = $d;
                }
            }

            $out[] = ['q' => $q, 'document_ids' => $ids ?: $allowed];
        }

        return $out;
    }

    // ------------------------------------------------------------ retrieval

    /**
     * One /retrieve call for all pending queries; merge results into $store by chunk key.
     *
     * @return int number of chunks not seen before
     */
    private function retrieveInto(array $queries, array $docIds, array &$store, array &$trace): int
    {
        $items = $this->rag->retrieve($queries, $docIds);

        $new = 0;
        foreach ($items as $item) {
            $key = $item['key'] ?? null;
            if (!is_string($key) || $key === '') {
                continue;
            }

            if (isset($store[$key])) {
                $store[$key]['score'] = max((float) $store[$key]['score'], (float) ($item['score'] ?? 0));
                continue;
            }

            $store[$key] = $item;
            $new++;
        }

        $trace[] = [
            'step' => 'retrieve',
            'queries' => count($queries),
            'returned' => count($items),
            'new' => $new,
        ];

        return $new;
    }

    /**
     * Top-K per document (so one document cannot crowd out the others), then ordered by
     * document and position so the context reads naturally.
     *
     * @return array<int, array<string, mixed>> a list (0-based keys)
     */
    private function select(array $store): array
    {
        $perDoc = max(1, (int) config('graphrag.agent.evidence_per_doc', 6));

        $byDoc = [];
        foreach ($store as $item) {
            $byDoc[$item['document_id']][] = $item;
        }

        $chosen = [];
        foreach ($byDoc as $items) {
            usort($items, fn ($a, $b) => ($b['score'] ?? 0) <=> ($a['score'] ?? 0));
            foreach (array_slice($items, 0, $perDoc) as $item) {
                $chosen[] = $item;
            }
        }

        usort($chosen, fn ($a, $b) => [$a['document_id'], $a['chunk_index'] ?? 0]
            <=> [$b['document_id'], $b['chunk_index'] ?? 0]);

        return array_values($chosen);
    }

    // -------------------------------------------------------------- grading

    /**
     * @return array<int, array{q: string, document_ids: array<int>}> follow-up queries;
     *         empty means the evidence is sufficient (or nothing more can be done)
     */
    private function grade(string $question, array $docIds, array $store, array $options, array &$trace): array
    {
        if (count($store) === 0) {
            $trace[] = ['step' => 'grade', 'sufficient' => false, 'reason' => 'no evidence'];

            return [];
        }

        $ranked = array_values($store);
        usort($ranked, fn ($a, $b) => ($b['score'] ?? 0) <=> ($a['score'] ?? 0));
        $ranked = array_slice($ranked, 0, 15);

        $snippet = (int) config('graphrag.agent.grade_snippet_chars', 500);
        $lines = [];
        foreach ($ranked as $e) {
            $lines[] = '(doc ' . $e['document_id'] . ', p.' . ($e['page'] ?? '?') . ') '
                . mb_substr((string) ($e['text'] ?? ''), 0, $snippet);
        }

        $prompt = strtr(self::GRADE_PROMPT, [
            '{question}' => $question,
            '{ids}' => json_encode($docIds),
            '{passages}' => implode("\n\n", $lines),
        ]);

        // On LLM failure, default to "sufficient" so the loop stops instead of spinning.
        $data = $this->llm->chatJson(
            [['role' => 'user', 'content' => $prompt]],
            ['sufficient' => true],
            $this->model($options)
        );

        $sufficient = $data['sufficient'] ?? true;
        if (is_string($sufficient)) {
            $sufficient = !in_array(strtolower(trim($sufficient)), ['false', 'no', '0'], true);
        }

        if ($sufficient) {
            $trace[] = ['step' => 'grade', 'sufficient' => true];

            return [];
        }

        $followUps = array_slice($this->parseQueries($data['missing'] ?? null, $docIds), 0, 3);
        $trace[] = ['step' => 'grade', 'sufficient' => false, 'follow_ups' => $followUps];

        return $followUps;
    }

    // ------------------------------------------------------------ synthesis

    private function systemPrompt(array $options): string
    {
        $language = $options['answer_language'] ?? null;

        $rule = is_string($language) && $language !== ''
            ? '(6) Answer ONLY in ' . $language . '.'
            : '(6) Answer in the same language as the question.';

        $extra = $options['extra_instructions'] ?? null;
        if (is_string($extra) && $extra !== '') {
            $rule .= ' ' . $extra;
        }

        return self::SYNTH_RULES . $rule;
    }

    private function synthesize(string $question, array $evidence, array $options): string
    {
        $sources = implode("\n\n", array_map(fn ($e) => $this->formatSource($e), $evidence));
        $user = $this->historyText($options)
            . "Sources:\n\n" . $sources . "\n\nQuestion: " . $question . "\n\nAnswer:";

        return trim($this->llm->chat([
            ['role' => 'system', 'content' => $this->systemPrompt($options)],
            ['role' => 'user', 'content' => $user],
        ], false, 0.1, $this->model($options)));
    }

    private function formatSource(array $e): string
    {
        $parts = ['document ' . $e['document_id']];
        if (($e['page'] ?? null) !== null) {
            $parts[] = 'page ' . $e['page'];
        }
        if (!empty($e['headings'])) {
            $parts[] = 'section: ' . implode(' > ', $e['headings']);
        }
        if (!empty($e['linked_via'])) {
            $parts[] = 'related through shared concepts: ' . implode(', ', $e['linked_via']);
        }

        $text = '[' . $e['sid'] . '] (' . implode(', ', $parts) . ")\n" . ($e['text'] ?? '');

        if (!empty($e['figures'])) {
            $lines = [];
            foreach ($e['figures'] as $f) {
                $name = $f['label'] ?? 'Unlabeled figure';
                $where = isset($f['page']) ? ' (page ' . $f['page'] . ')' : '';
                $caption = !empty($f['caption']) ? ': ' . mb_substr((string) $f['caption'], 0, 200) : '';
                $lines[] = '  - ' . $name . $where . $caption;
            }
            $text .= "\nLinked figures:\n" . implode("\n", $lines);
        }

        return $text;
    }

    /**
     * Shape expected by POST /verify.
     *
     * @return array<int, array{sid: string, text: string, figure_numbers: array<string>}>
     */
    private function toSources(array $evidence): array
    {
        $sources = [];
        foreach ($evidence as $e) {
            $numbers = [];
            foreach (($e['figures'] ?? []) as $f) {
                if (isset($f['number']) && $f['number'] !== '') {
                    $numbers[] = (string) $f['number'];
                }
            }

            $sources[] = [
                'sid' => $e['sid'],
                'text' => (string) ($e['text'] ?? ''),
                'figure_numbers' => $numbers,
            ];
        }

        return $sources;
    }
}