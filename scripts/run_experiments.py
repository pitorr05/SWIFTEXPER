"""Sequential H100 experiments: CNN/DM, HumanEval pass@1, and optional GSM8K accuracy."""
import argparse
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import zipfile

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / 'runtime'))
from research_utils import METRIC_PROTOCOL, aggregate, compare_aggregates, read_records, validate_pair, write_json
from source_patches import prepare
from gsm8k import score_records, compare_scores

def note(message):
    """Display progress immediately; never write runtime messages to files."""
    print(message, flush=True)


def git(*arguments):
    return subprocess.check_output(['git', '-C', str(REPO), *arguments], stderr=subprocess.PIPE)


def export_source(ref, destination, label, snapshots):
    try:
        commit = git('rev-parse', '--verify', f'{ref}^{{commit}}').decode().strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Cannot resolve {ref!r}. Fetch the branch or set SWIFT_REF / ADAPTIVE_REF.') from error
    archive = git('archive', '--format=zip', commit)
    archive_path = snapshots / f'{label}.original.zip'
    archive_path.write_bytes(archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('Unsafe Git archive member')
            if '__pycache__' not in path.parts and path.suffix != '.pyc':
                bundle.extract(member, destination)
    info = {'ref': ref, 'commit': commit, 'original_archive': str(archive_path),
            'archive_sha256': hashlib.sha256(archive).hexdigest()}
    write_json(destination / 'SOURCE_PROVENANCE.json', info)
    return info


def code_sources(folder):
    return {p.relative_to(folder).as_posix(): p.read_text(encoding='utf-8')
            for sub in ('evaluation_llama', 'model/swift') for p in sorted((folder / sub).glob('*.py'))}


def save_changes(folder, original):
    patched = code_sources(folder)
    lines = []
    for path in sorted(set(original) | set(patched)):
        lines.extend(difflib.unified_diff(original.get(path, '').splitlines(keepends=True),
                                        patched.get(path, '').splitlines(keepends=True),
                                        fromfile=f'original/{path}', tofile=f'run/{path}'))
    (folder / 'SOURCE_CHANGES.patch').write_text(''.join(lines), encoding='utf-8')
    for path, content in patched.items():
        compile(content, str(folder / path), 'exec')


def run(command, cwd):
    note(f'RUN {shlex.join(map(str, command))}')
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        # Inherit stdout/stderr so downloads, tqdm, optimization and tracebacks stay live.
        subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Command failed (exit {error.returncode}): {shlex.join(map(str, command))}. '
                           'The original subprocess error is printed directly above.') from error


def check_environment(output):
    packages = {}
    missing = []
    for name in ('torch', 'transformers', 'accelerate', 'fschat', 'datasets', 'bayesian-optimization', 'human-eval'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    write_json(output / 'environment.json', {'python': sys.version, 'packages': packages,
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    if missing:
        raise RuntimeError(f'Missing dependencies in the active Python environment: {", ".join(missing)}. '
                           'Activate the existing SWIFT environment; install human-eval if missing.')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA GPU must be visible; set CUDA_VISIBLE_DEVICES=0')
    if packages['transformers'] != '4.37.1':
        note('NOTE: original SWIFT pins transformers==4.37.1; active version is '
             + packages['transformers'] + '. No packages were changed.')
    run(['nvidia-smi'], REPO)


def compare_task(folder, task, expected, write_comparison=True):
    methods = ('baseline', 'swift', 'adaptive')
    records = {method: read_records(folder / f'{task}.{method}.jsonl') for method in methods}
    manifests = []
    metadata = []
    for method in methods:
        path = folder / f'{task}.{method}.jsonl'
        summary = json.loads(Path(str(path) + '.summary.json').read_text(encoding='utf-8'))
        if len(records[method]) != expected or summary.get('status') != 'complete' or summary['samples'] != expected:
            raise RuntimeError(f'Incomplete inference: {path}')
        manifests.append(json.loads(Path(str(path) + '.manifest.json').read_text(encoding='utf-8')))
        metadata.append(json.loads(Path(str(path) + '.metadata.json').read_text(encoding='utf-8')))
    if manifests[1:] != [manifests[0], manifests[0]]:
        raise RuntimeError('Dataset ID order or one-shot prompts differ between methods')
    for candidate in metadata[1:]:
        if candidate['model_revision'] != metadata[0]['model_revision']:
            raise RuntimeError('Loaded target model revisions differ')
        for key in ('model_path', 'model_revision', 'dtype', 'seed', 'data_seed', 'temperature',
                    'top_p', 'max_new_tokens', 'data_num', 'task_name'):
            if candidate['config'][key] != metadata[0]['config'][key]:
                raise RuntimeError(f'Inference configurations differ: {key}')
    summaries = {method: aggregate(records[method]) for method in methods}
    comparisons = {}
    for method, base in (('swift', 'baseline'), ('adaptive', 'baseline'), ('adaptive', 'swift')):
        validate_pair(records[method], records[base])
        mismatches = [r['task_id'] for r, b in zip(records[method], records[base])
                      if r['output_token_ids'] != b['output_token_ids']]
        comparisons[f'{method}_vs_{base}'] = {
            **compare_aggregates(summaries[method], summaries[base]),
            'token_mismatch_task_ids': mismatches, 'token_mismatch_count': len(mismatches)}
        if mismatches:
            note(f'NOTE: {task}/{method} vs {base}: {len(mismatches)} token mismatches; retained for analysis.')
    if write_comparison:
        write_json(folder / f'{task}.comparisons.json', comparisons)
    return summaries, comparisons


FIELDS = ['metric_protocol', 'seed', 'dataset', 'model', 'method', 'samples', 'tokens', 'inference_seconds',
          'tokens_per_second', 'speedup_vs_baseline', 'speedup_vs_swift',
          'global_tokens_per_second', 'global_throughput_speedup_vs_baseline', 'global_throughput_speedup_vs_swift',
          'inference_time_speedup_vs_baseline', 'inference_time_speedup_vs_swift', 'mean_accepted_length',
          'acceptance_rate', 'cache_hit_rate', 'embedding_seconds', 'search_seconds',
          'token_mismatch_vs_baseline', 'pass_at_1', 'passed', 'evaluated', 'pass_at_1_delta_vs_baseline',
          'accuracy', 'accuracy_delta_vs_baseline', 'correct', 'missing_answer_rate', 'token_limit_rate',
          'correct_to_wrong_vs_baseline', 'wrong_to_correct_vs_baseline']


def write_results(output, rows):
    target = output / 'results.csv'
    temporary = output / 'results.csv.tmp'
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def build_result_rows(seed, task, model, summaries, comparisons, scores):
    rows = []
    for method in ('baseline', 'swift', 'adaptive'):
        summary = summaries[method]
        baseline_ratio = compare_aggregates(summary, summaries['baseline'])
        swift_ratio = compare_aggregates(summary, summaries['swift'])
        row = {key: summary[key] for key in ('metric_protocol', 'samples', 'tokens', 'inference_seconds',
               'tokens_per_second', 'global_tokens_per_second', 'mean_accepted_length', 'acceptance_rate',
               'cache_hit_rate', 'embedding_seconds', 'search_seconds')}
        row.update(seed=seed, dataset=task, model=model, method=method,
                   speedup_vs_baseline=baseline_ratio['speedup_ratio'],
                   speedup_vs_swift=swift_ratio['speedup_ratio'],
                   global_throughput_speedup_vs_baseline=baseline_ratio['global_throughput_speedup'],
                   global_throughput_speedup_vs_swift=swift_ratio['global_throughput_speedup'],
                   inference_time_speedup_vs_baseline=baseline_ratio['inference_time_speedup'],
                   inference_time_speedup_vs_swift=swift_ratio['inference_time_speedup'],
                   token_mismatch_vs_baseline=0 if method == 'baseline' else
                   comparisons[f'{method}_vs_baseline']['token_mismatch_count'])
        if method == 'baseline':
            row['acceptance_rate'] = None
        if scores and task.startswith('gsm8k.'):
            score = scores[method]
            paired = compare_scores(score, scores['baseline'])
            row.update(accuracy=score['accuracy'], correct=score['correct'], evaluated=score['total'],
                       accuracy_delta_vs_baseline=score['accuracy']-scores['baseline']['accuracy'],
                       missing_answer_rate=score['missing_answer_rate'], token_limit_rate=score['token_limit_rate'],
                       correct_to_wrong_vs_baseline=len(paired['correct_to_wrong_ids']),
                       wrong_to_correct_vs_baseline=len(paired['wrong_to_correct_ids']))
        elif scores:
            row.update(pass_at_1=scores[method]['pass@1'], passed=scores[method]['passed'], evaluated=164,
                       pass_at_1_delta_vs_baseline=scores[method]['pass@1']-scores['baseline']['pass@1'])
        rows.append(row)
    return rows


def task_configs(args):
    """Artifact key, actual task, model, model ID, count, budget, top-p, revision."""
    llama_id = Path(args.llama_model.rstrip('/\\')).name
    code_id = Path(args.code_model.rstrip('/\\')).name
    for task in args.tasks:
        if task == 'cnndm':
            yield task, task, args.llama_model, llama_id, 1000, 64, 0.85, args.llama_revision
        elif task == 'humaneval':
            yield task, task, args.code_model, code_id, 164, 512, 0.95, args.code_revision
        else:
            for profile in args.gsm8k_profiles:
                limit = {'paper64': 64, 'quality512': 512}[profile]
                yield f'gsm8k.{profile}', task, args.llama_model, llama_id, 1000, limit, 0.85, args.llama_revision


def execute(args, output, reference, adaptive):
    check_environment(output)
    rows = []
    for repetition, seed in enumerate(args.seeds):
        folder = output / f'seed_{seed}'
        folder.mkdir()
        methods = ['baseline', 'swift', 'adaptive']
        order = methods[repetition % 3:] + methods[:repetition % 3]
        for task, inference_task, model, model_id, count, limit, top_p, revision in task_configs(args):
            note(f'Run {output.name}, seed {seed}: starting {task}, {count} tasks; order: {", ".join(order)}.')
            for method in order:
                root = adaptive if method == 'adaptive' else reference
                module = 'inference_baseline' if method == 'baseline' else 'inference_swift'
                command = [sys.executable, '-m', f'evaluation_llama.{module}', '--model-path', model,
                           '--model-id', model_id, '--task-name', inference_task, '--data-num', str(count),
                           '--max-new-tokens', str(limit), '--temperature', '0.0', '--top-p', str(top_p),
                           '--dtype', 'float16', '--seed', str(seed), '--data-seed', str(args.data_seed),
                           '--answer-file', str(folder / f'{task}.{method}.jsonl')]
                if revision:
                    command += ['--model-revision', revision]
                if method != 'baseline':
                    command += ['--skip-ratio', '0.45', '--context-window', '32', '--opt-interval', '1',
                                '--bayes-interval', '25', '--max-opt-iter', '1000', '--max-tolerance-iter', '300',
                                '--max-score', '0.95', '--optimization', '--bayes']
                if method == 'adaptive':
                    command += ['--cache-enabled', '--cache-clear', '--cache-threshold', '0.85', '--cache-max-size', '100']
                run(command, root)
                note(f'Inference complete: {task}/{method}, seed {seed}; output: {folder / (task+"."+method+".jsonl")}. '
                     + ('HumanEval pass@1 has not been evaluated yet.' if task == 'humaneval' else ''))
            summaries, comparisons = compare_task(folder, task, count)
            scores = {}
            if task == 'humaneval':
                for method in methods:
                    score_path = folder / f'humaneval.{method}.pass1.json'
                    run([sys.executable, str(SCRIPTS / 'evaluate_humaneval.py'),
                         '--input', str(folder / f'humaneval.{method}.jsonl'),
                         '--samples', str(folder / f'humaneval.{method}.samples.jsonl'),
                         '--score', str(score_path), '--workers', str(args.eval_workers),
                         '--timeout', str(args.eval_timeout)], reference)
                    scores[method] = json.loads(score_path.read_text(encoding='utf-8'))
                    note(f'HumanEval {method}, seed {seed}: pass@1 = {scores[method]["pass@1"]:.6f}, '
                         f'{scores[method]["passed"]}/164 passed. Results: {score_path}')
            elif inference_task == 'gsm8k':
                for method in methods:
                    scores[method] = score_records(read_records(folder / f'{task}.{method}.jsonl'), count)
                for method in methods:
                    score = scores[method]
                    score['vs_baseline'] = compare_scores(score, scores['baseline'])
                    write_json(folder / f'{task}.{method}.accuracy.json', score)
                    note(f'{task}/{method}: accuracy={score["accuracy"]:.4f}, '
                         f'missing={score["missing_answer_rate"]:.4f}, token_limit={score["token_limit_rate"]:.4f}')
            rows.extend(build_result_rows(seed, task, model, summaries, comparisons, scores))
            write_results(output, rows)
            note(f'Results saved for {task}, seed {seed}: {output / "results.csv"}. '
                 'Token mismatches are reported separately from functional correctness.')
    return rows


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, default=os.environ.get('OUTPUT_DIR'))
    parser.add_argument('--swift-ref', default=os.environ.get('SWIFT_REF', 'main'))
    parser.add_argument('--adaptive-ref', default=os.environ.get('ADAPTIVE_REF', 'origin/AdaptiveCache'))
    parser.add_argument('--llama-model', default=os.environ.get('LLAMA_MODEL_PATH', 'meta-llama/Llama-2-13b-hf'))
    parser.add_argument('--code-model', default=os.environ.get('CODE_MODEL_PATH', 'codellama/CodeLlama-13b-hf'))
    parser.add_argument('--llama-revision', default=os.environ.get('LLAMA_REVISION'))
    parser.add_argument('--code-revision', default=os.environ.get('CODE_REVISION'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[int(x) for x in os.environ.get('SEEDS', '2024').replace(',', ' ').split()])
    parser.add_argument('--data-seed', type=int, default=2024)
    parser.add_argument('--tasks', nargs='+', choices=['cnndm', 'humaneval', 'gsm8k'],
                        default=['cnndm', 'humaneval'])
    parser.add_argument('--gsm8k-profiles', nargs='+', choices=['paper64', 'quality512'],
                        default=['paper64', 'quality512'])
    parser.add_argument('--eval-workers', type=int, default=4)
    parser.add_argument('--eval-timeout', type=float, default=3.0)
    parser.add_argument('--prepare-only', action='store_true', help='Export sources and patches only; no GPU inference or scoring')
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.gsm8k_profiles)) != len(args.gsm8k_profiles):
        parser.error('Duplicate tasks or GSM8K profiles are not supported')
    import math
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.eval_workers < 1 or not math.isfinite(args.eval_timeout) or args.eval_timeout <= 0:
        parser.error('Distinct seeds, positive eval-workers and eval-timeout required')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output_dir or REPO / '_local' / 'runs' / f'h100_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = 'prepare sources'
    try:
        note(f'Starting run: {output}. Source refs: {args.swift_ref}, {args.adaptive_ref}. Original working source is unchanged.')
        snapshots = output / 'snapshots'
        snapshots.mkdir()
        reference, adaptive = output / 'sources/swift', output / 'sources/adaptive'
        reference.mkdir(parents=True)
        adaptive.mkdir(parents=True)
        origins = {'swift': export_source(args.swift_ref, reference, 'swift', snapshots),
                   'adaptive': export_source(args.adaptive_ref, adaptive, 'adaptive', snapshots)}
        before = {str(folder): code_sources(folder) for folder in (reference, adaptive)}
        prepare(reference, adaptive, SCRIPTS / 'runtime')
        for folder in (reference, adaptive):
            save_changes(folder, before[str(folder)])
        # Preserve the exact orchestrator and patch assets used by this run.
        bundle = output / 'driver'
        shutil.copytree(SCRIPTS, bundle, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(output_dir=str(output), sources=origins, warmup=False, metric_protocol=METRIC_PROTOCOL,
                      human_eval_completion_policy='prompt_context_decode_v1; no stripping or code repair')
        write_json(output / 'run_config.json', config)
        note('Sources prepared; original ZIP, SOURCE_CHANGES.patch and configuration saved. '
             f'SWIFT {origins["swift"]["commit"]}, AdaptiveCache {origins["adaptive"]["commit"]}. No smoke test or warm-up.')
        if args.prepare_only:
            write_json(output / 'status.json', {'status': 'prepared_only', 'inference_ran': False, 'pass1_evaluated': False})
            note(f'Sources prepared only: {output}. No GPU inference or evaluation was run.')
            return
        stage = 'inference / evaluation'
        rows = execute(args, output, reference, adaptive)
        write_json(output / 'status.json', {'status': 'complete', 'rows': len(rows),
                   'pass1_evaluated': 'humaneval' in args.tasks, 'gsm8k_evaluated': 'gsm8k' in args.tasks})
        table = '\n'.join(f'- seed {r["seed"]}, {r["dataset"]}/{r["method"]}: {r["tokens_per_second"]:.3f} tokens/s, '
                          f'speedup vs baseline {r["speedup_vs_baseline"]:.3f}x'
                          + (f', pass@1 {r["pass_at_1"]:.6f} ({r["passed"]}/164)' if 'pass_at_1' in r else '')
                          + (f', accuracy {r["accuracy"]:.4f}, token_limit {r["token_limit_rate"]:.4f}' if 'accuracy' in r else '') for r in rows)
        note(f'Run complete: {output / "results.csv"}\n\n{table}\n\n'
             'Matching pass@1 alone does not establish preservation of the full output distribution.')
        print(f'\nDONE: {output / "results.csv"}', flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'failed', 'stage': stage, 'error': repr(error)})
        note(f'Run failed: {output}; stage: {stage}; error: {error}. Partial artifacts retained; run is incomplete.')
        raise


if __name__ == '__main__':
    main()
