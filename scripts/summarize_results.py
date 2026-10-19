"""Recompute SWIFT-author speed metrics from an existing run; no inference or code execution."""
import argparse
import csv
import json
import math
from pathlib import Path

from run_experiments import FIELDS, METRIC_PROTOCOL, build_result_rows, compare_task, note, write_json


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
    for seed in config['seeds']:
        folder = run_dir / f'seed_{seed}'
        for task, count, model in (('cnndm', 1000, config['llama_model']),
                                   ('humaneval', 164, config['code_model'])):
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
            rows.extend(build_result_rows(seed, task, model, summaries, comparisons, scores))
            datasets.append({'seed': seed, 'dataset': task, 'summaries': summaries, 'comparisons': comparisons})
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
         f'{csv_path}, {json_path}. No inference or evaluation was run; existing artifacts are unchanged.')


if __name__ == '__main__':
    main()
