"""Re-decode and re-score saved HumanEval tokens on CPU; original artifacts stay intact."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
from completion_decode import POLICY, decode_continuation
from research_utils import digest, read_records, validate_pair, write_json
from evaluate_humaneval import score_samples

METHODS = ('baseline', 'swift', 'adaptive')


def syntax_error(prompt, completion):
    try:
        compile(prompt + completion, '<HumanEval diagnostic>', 'exec')
        return None
    except (SyntaxError, ValueError, OverflowError) as error:
        return f'{type(error).__name__}: {error}'


def rebuild(records, tokenizer, problems):
    if (len(records) != 164 or {r['task_id'] for r in records} != set(problems)
            or any(r.get('task_name') != 'humaneval' for r in records)):
        raise ValueError('Expected the complete standard 164 HumanEval tasks')
    samples, diagnostics = [], []
    for row in records:
        task_id = row['task_id']
        prompt = problems[task_id]['prompt']
        inputs, outputs = row['input_token_ids'], row['output_token_ids']
        if not inputs or any(type(i) is not int or i < 0 for i in inputs + outputs):
            raise ValueError(f'Invalid token IDs: {task_id}')
        if row['input_sha256'] != digest(inputs):
            raise ValueError(f'Saved input hash mismatch: {task_id}')
        text = decode_continuation(tokenizer, inputs, outputs, expected_prompt=prompt)
        saved = row['choices'][0]['turns']
        samples.append({'task_id': task_id, 'completion': text})
        diagnostics.append({'task_id': task_id, 'changed': text != saved,
                            'saved_first_line': saved.split('\n', 1)[0],
                            'context_first_line': text.split('\n', 1)[0],
                            'saved_syntax_error': syntax_error(prompt, saved),
                            'context_syntax_error': syntax_error(prompt, text)})
    return samples, diagnostics


def recheck(run_dir, workers=4, timeout=3.0, diagnose_only=False, allow_download=False):
    from transformers import AutoTokenizer
    from human_eval.data import read_problems
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / 'run_config.json').read_text(encoding='utf-8'))
    if config.get('prepare_only'):
        raise ValueError('This run contains prepared sources only')
    problems = read_problems()
    if len(problems) != 164:
        raise ValueError('Installed evaluator does not contain the standard 164 tasks')
    output = run_dir / ('humaneval_context_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    rows = []
    try:
        write_json(output / 'provenance.json', {
            'source_run': str(run_dir), 'policy': POLICY, 'inference_ran': False,
            'diagnose_only': diagnose_only, 'problems_sha256': digest(problems),
            'decode_source_sha256': digest(inspect.getsource(decode_continuation)),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('transformers', 'tokenizers', 'human-eval')},
            'note': 'No indentation repair, cropping, stop-word filtering or changes to saved tokens/timing'})
        for seed in config['seeds']:
            source_folder = run_dir / f'seed_{seed}'
            folder = output / f'seed_{seed}'
            folder.mkdir()
            records, metadata = {}, {}
            for method in METHODS:
                path = source_folder / f'humaneval.{method}.jsonl'
                records[method] = read_records(path)
                summary = json.loads(Path(str(path) + '.summary.json').read_text(encoding='utf-8'))
                if summary.get('status') != 'complete' or summary.get('samples') != 164:
                    raise ValueError(f'Incomplete source inference: {path}')
                metadata[method] = json.loads(Path(str(path) + '.metadata.json').read_text(encoding='utf-8'))
                validate_pair(records[method], records['baseline'])
            identity = metadata['baseline']
            active_version = importlib.metadata.version('transformers')
            for candidate in metadata.values():
                original_version = candidate.get('packages', {}).get('transformers')
                if original_version and original_version != active_version:
                    raise ValueError(f'Use the original transformers version {original_version}, '
                                     f'not {active_version}, to re-decode saved tokens')
            model = identity['config']['model_path']
            revision = identity.get('model_revision') or identity['config'].get('model_revision')
            for candidate in metadata.values():
                if (candidate['config']['model_path'] != model
                        or (candidate.get('model_revision') or candidate['config'].get('model_revision')) != revision):
                    raise ValueError('Target model path/revision differs between methods')
            print(f'CPU tokenizer: {model}, revision={revision}; seed {seed}', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                                      local_files_only=not allow_download)
            if tokenizer.vocab_size < 1000:
                raise ValueError('Tokenizer vocabulary is incomplete; check cached tokenizer files')
            tokenizer_info = {'model': model, 'revision': revision, 'class': type(tokenizer).__name__,
                              'vocab_sha256': digest(tokenizer.get_vocab())}
            if getattr(tokenizer, 'is_fast', False):
                tokenizer_info['backend_sha256'] = digest(tokenizer.backend_tokenizer.to_str())
            write_json(folder / 'tokenizer.json', tokenizer_info)
            # Validate and reconstruct all three methods before executing any generated code.
            rebuilt = {m: rebuild(records[m], tokenizer, problems) for m in METHODS}
            for method in METHODS:
                samples, diagnostics = rebuilt[method]
                samples_path = folder / f'humaneval.{method}.samples.jsonl'
                with open(samples_path, 'x', encoding='utf-8') as stream:
                    for sample in samples:
                        stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
                source = source_folder / f'humaneval.{method}.jsonl'
                report = {'policy': POLICY, 'source': str(source),
                          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                          'metadata_sha256': hashlib.sha256(Path(str(source)+'.metadata.json').read_bytes()).hexdigest(),
                          'changed': sum(d['changed'] for d in diagnostics),
                          'saved_syntax_errors': sum(d['saved_syntax_error'] is not None for d in diagnostics),
                          'context_syntax_errors': sum(d['context_syntax_error'] is not None for d in diagnostics),
                          'details': diagnostics}
                write_json(folder / f'humaneval.{method}.diagnostics.json', report)
                print(f'{method}: changed {report["changed"]}/164; syntax errors '
                      f'{report["saved_syntax_errors"]} -> {report["context_syntax_errors"]}', flush=True)
                old_path = source_folder / f'humaneval.{method}.pass1.json'
                old = json.loads(old_path.read_text(encoding='utf-8')) if old_path.exists() else {}
                row = {'seed': seed, 'model': model, 'method': method, 'policy': POLICY,
                       'old_pass_at_1': old.get('pass@1'), 'changed_completions': report['changed'],
                       'saved_syntax_errors': report['saved_syntax_errors'],
                       'context_syntax_errors': report['context_syntax_errors']}
                if not diagnose_only:
                    score = score_samples(samples_path, folder / f'humaneval.{method}.pass1.json',
                                          workers, timeout, completion_policy=POLICY)
                    row.update(pass_at_1=score['pass@1'], passed=score['passed'], total=164)
                rows.append(row)
        if not rows:
            raise ValueError('No seeds to re-evaluate')
        with open(output / 'results.csv', 'x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / 'status.json', {'status': 'diagnosed' if diagnose_only else 'complete',
                   'policy': POLICY, 'inference_ran': False, 'rows': len(rows)})
        print(f'DONE: {output / "results.csv"}. Original results and timing are unchanged.', flush=True)
        return output
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'failed', 'error': repr(error)})
        print(f'Recheck failed; partial artifacts: {output}', flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=3.0)
    parser.add_argument('--diagnose-only', action='store_true', help='Decode and compile only; do not execute tests')
    parser.add_argument('--allow-download', action='store_true', help='Allow downloading tokenizer files only')
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('workers and timeout must be finite and positive')
    for folder in args.run_dir:
        recheck(folder, args.workers, args.timeout, args.diagnose_only, args.allow_download)


if __name__ == '__main__':
    main()
