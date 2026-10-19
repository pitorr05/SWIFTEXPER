"""Small, dependency-free helpers shared by both research checkouts."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def validate_output_path(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path, Path(str(path) + '.metadata.json'), Path(str(path) + '.manifest.json'),
                      Path(str(path) + '.summary.json')):
        if candidate.exists():
            raise FileExistsError(f'Refusing to reuse run artifact: {candidate}')
    return str(path)


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def read_records(path):
    with open(path, encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or any('choices' not in row or 'task_id' not in row for row in records):
        raise ValueError(f'{path}: expected nonempty per-task records with task_id and choices')
    ids = [row['task_id'] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path}: duplicate task IDs; concatenated runs are not supported')
    return records


def validate_pair(records, baseline):
    if [r['task_id'] for r in records] != [r['task_id'] for r in baseline]:
        raise ValueError('Task IDs or their order differ')
    for left, right in zip(records, baseline):
        if left.get('task_name') != right.get('task_name'):
            raise ValueError('Task names differ')
        if not left.get('input_sha256') or left['input_sha256'] != right.get('input_sha256'):
            raise ValueError(f'Effective prompts differ for {left["task_id"]}')


def export_humaneval(source, destination, expected_count=164, expected_ids=None):
    records = read_records(source)
    if len(records) != expected_count or any(r.get('task_name') != 'humaneval' for r in records):
        raise ValueError(f'Expected {expected_count} HumanEval tasks')
    if expected_ids is not None and set(r['task_id'] for r in records) != set(expected_ids):
        raise ValueError('HumanEval task IDs do not match the evaluator problem set')
    # No indentation stripping, markdown cleanup or answer-dependent repair.
    with open(destination, 'x', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps({'task_id': row['task_id'],
                                     'completion': row['choices'][0]['turns']}, ensure_ascii=False) + '\n')


def aggregate(records):
    choices = [r['choices'][0] for r in records]
    times = [sum(c['wall_time']) for c in choices]
    counts = [sum(c['new_tokens']) for c in choices]
    if any(t <= 0 for t in times):
        raise ValueError('All inference times must be positive')
    lengths = [n for c in choices for n in c['accept_lengths']]
    draft = sum(c.get('draft_token_num', 0) for c in choices)
    accepted = sum(c.get('accepted_draft_tokens', 0) for c in choices)
    cache = [r.get('forward_metrics', {}) for r in records]
    attempts = [m for m in cache if m.get('cache_enabled')]
    return {
        'samples': len(records), 'tokens': sum(counts), 'inference_seconds': sum(times),
        'tokens_per_second': sum(counts) / sum(times),
        'mean_sample_tokens_per_second': sum(n/t for n, t in zip(counts, times)) / len(times),
        'mean_accepted_length': sum(lengths) / len(lengths) if lengths else None,
        'draft_token_num': draft, 'accepted_draft_tokens': accepted,
        'acceptance_rate': accepted / draft if draft else None,
        'cache_requests': len(attempts), 'cache_hits': sum(bool(m.get('cache_hit')) for m in attempts),
        'cache_hit_rate': sum(bool(m.get('cache_hit')) for m in attempts) / len(attempts) if attempts else None,
        'embedding_seconds': sum(m.get('embedding_seconds', 0) for m in cache),
        'search_seconds': sum(m.get('search_seconds', 0) for m in cache),
        'cache_update_seconds': sum(m.get('cache_update_seconds', 0) for m in cache),
        'optimization_host_seconds': sum(m.get('optimization_host_seconds', 0) for m in cache),
        'optimization_gpu_ms': sum(m.get('optimization_gpu_ms', 0) for m in cache),
    }
