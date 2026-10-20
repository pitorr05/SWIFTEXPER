"""Export and score one completion per HumanEval task, producing machine-readable pass@1."""
import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'assets'))
from research_utils import export_humaneval, read_records, write_json


def evaluate(source, samples, score_file, workers=4, timeout=3.0):
    from human_eval.data import read_problems
    problems = read_problems()
    policies = {row.get('completion_policy', 'legacy_saved_continuation') for row in read_records(source)}
    if len(policies) != 1:
        raise ValueError('Mixed HumanEval completion policies in one file')
    export_humaneval(source, samples, expected_count=164, expected_ids=problems)
    return score_samples(samples, score_file, workers, timeout, completion_policy=policies.pop())


def score_samples(samples, score_file, workers=4, timeout=3.0,
                  completion_policy='saved continuation; no stripping or code repair'):
    """Score an existing samples export without altering any completion."""
    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness
    import human_eval.execution as execution
    engine_source = inspect.getsource(execution)
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'exec'
               for node in ast.walk(ast.parse(engine_source))):
        raise RuntimeError('HumanEval execution is disabled in the installed evaluator; enable its exec call before scoring. '
                           'Refusing to report pass@1 without executing the generated functions.')
    problems = read_problems()
    if len(problems) != 164:
        raise ValueError(f'Expected the standard 164 HumanEval tasks, found {len(problems)}')
    score_file = Path(score_file)
    results_path = Path(str(samples) + '_results.jsonl')
    if score_file.exists() or results_path.exists():
        raise FileExistsError('Refusing to overwrite existing HumanEval scores or evaluator results')
    with open(samples, encoding='utf-8') as stream:
        exported = [json.loads(line) for line in stream if line.strip()]
    ids = [row['task_id'] for row in exported]
    if len(ids) != 164 or len(set(ids)) != 164 or set(ids) != set(problems):
        raise ValueError('Samples must contain exactly the standard 164 unique HumanEval tasks')
    scores = evaluate_functional_correctness(str(samples), k=[1], n_workers=workers, timeout=timeout)
    if 'pass@1' not in scores or not math.isfinite(float(scores['pass@1'])):
        raise RuntimeError('HumanEval evaluator did not return a finite pass@1 score')
    results_path = Path(str(samples) + '_results.jsonl')
    with open(results_path, encoding='utf-8') as stream:
        results = [json.loads(line) for line in stream if line.strip()]
    ids = [row['task_id'] for row in results]
    if len(results) != 164 or len(set(ids)) != 164 or set(ids) != set(problems):
        raise RuntimeError('Evaluator results are missing, duplicated or from a different problem set')
    if any(type(row.get('passed')) is not bool for row in results):
        raise RuntimeError('Evaluator result is missing a boolean passed field')
    passed = sum(row['passed'] for row in results)
    pass1 = float(scores['pass@1'])
    if not 0 <= pass1 <= 1 or not math.isclose(pass1, passed / 164, abs_tol=1e-12):
        raise RuntimeError('pass@1 does not agree with per-task results')
    score = {'status': 'complete', 'pass@1': pass1, 'passed': passed, 'total': 164,
             'results_file': str(results_path), 'workers': workers, 'timeout_seconds': timeout,
             'completion_policy': completion_policy,
             'execution_source_sha256': hashlib.sha256(engine_source.encode('utf-8')).hexdigest()}
    write_json(score_file, score)
    print(json.dumps(score, indent=2), flush=True)
    return score


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--score', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=3.0)
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('workers and timeout must be positive and finite')
    evaluate(args.input, args.samples, args.score, args.workers, args.timeout)


if __name__ == '__main__':
    main()
