"""Recompute speed metrics and optional GSM8K numeric accuracy; no inference or code execution."""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

from run_experiments import (FIELDS, METRIC_PROTOCOL, build_result_rows, compare_task, note, write_json,
                             task_configs, read_records, score_records)


def summarize(run_dir, output_dir=None):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else run_dir
    csv_path = output_dir / f'results.{METRIC_PROTOCOL}.csv'
    json_path = output_dir / f'metrics.{METRIC_PROTOCOL}.json'
    if csv_path.exists() or json_path.exists():
        raise FileExistsError('Versioned results already exist; select a new --output-dir')
    config = json.loads((run_dir / 'run_config.json').read_text(encoding='utf-8'))
    if config.get('prepare_only'):
        raise ValueError('This run only prepared sources; there are no inference results to summarize')
    rows, datasets = [], []
    task_args = SimpleNamespace(**{**config, 'tasks': config.get('tasks', ['cnndm', 'humaneval']),
                                  'gsm8k_profiles': config.get('gsm8k_profiles', ['paper64', 'quality512']),
                                  'llama_revision': config.get('llama_revision'),
                                  'code_revision': config.get('code_revision')})
    for seed in config['seeds']:
        folder = run_dir / f'seed_{seed}'
        for task, inference_task, model, _, count, _, _, _ in task_configs(task_args):
            summaries, comparisons = compare_task(folder, task, count, write_comparison=False)
            scores = {}
            if task == 'humaneval':
                for method in ('baseline', 'swift', 'adaptive'):
                    score_path = folder / f'humaneval.{method}.pass1.json'
                    score = json.loads(score_path.read_text(encoding='utf-8'))
                    value, passed = score.get('pass@1'), score.get('passed')
                    if (score.get('status') != 'complete' or score.get('total') != 164
                            or type(passed) is not int or not 0 <= passed <= 164
                            or not isinstance(value, (int, float)) or not math.isfinite(value)
                            or not math.isclose(value, passed/164, abs_tol=1e-12)):
                        raise ValueError(f'Invalid existing pass@1 result: {score_path}')
                    scores[method] = score
            elif inference_task == 'gsm8k':
                # Pure numeric scoring from immutable outputs; no model or code execution.
                for method in ('baseline', 'swift', 'adaptive'):
                    scores[method] = score_records(read_records(folder / f'{task}.{method}.jsonl'), count)
            rows.extend(build_result_rows(seed, task, model, summaries, comparisons, scores))
            datasets.append({'seed': seed, 'dataset': task, 'summaries': summaries, 'comparisons': comparisons,
                             'gsm8k_scores': scores if inference_task == 'gsm8k' else None})
    if not rows:
        raise ValueError('Run configuration has no seeds to summarize')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, {'metric_protocol': METRIC_PROTOCOL, 'source_run': str(run_dir),
                          'source_metric_protocol': config.get('metric_protocol', 'legacy_global_throughput_v0'),
                          'inference_ran': False, 'pass1_recomputed': False, 'datasets': datasets})
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    csv_path, json_path = summarize(args.run_dir, args.output_dir)
    note(f'Speed metrics recomputed from {args.run_dir.resolve()} using {METRIC_PROTOCOL}: '
         f'{csv_path}, {json_path}. No GPU inference or generated-code execution was run; '
         'GSM8K numeric scores are recomputed when selected. Existing artifacts are unchanged.')


if __name__ == '__main__':
    main()
