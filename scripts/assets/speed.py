"""Compare aligned research artifacts without assuming 1,000 records."""
import argparse
import json
from evaluation_llama.research_utils import aggregate, read_records, validate_pair, write_json


def compare(file_path, base_path, expected_count=None):
    rows, baseline = read_records(file_path), read_records(base_path)
    validate_pair(rows, baseline)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f'Expected {expected_count} rows, found {len(rows)}')
    result, base = aggregate(rows), aggregate(baseline)
    result['baseline'] = base
    result['throughput_speedup'] = result['tokens_per_second'] / base['tokens_per_second']
    result['inference_time_speedup'] = base['inference_seconds'] / result['inference_seconds']
    result['mean_sample_throughput_ratio'] = result['mean_sample_tokens_per_second'] / base['mean_sample_tokens_per_second']
    mismatches = [r['task_id'] for r, b in zip(rows, baseline) if r['output_token_ids'] != b['output_token_ids']]
    result['token_mismatch_task_ids'] = mismatches
    result['exact_token_agreement'] = (len(rows)-len(mismatches)) / len(rows)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--file-path', required=True)
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--output')
    parser.add_argument('--require-token-match', action='store_true')
    args = parser.parse_args()
    result = compare(args.file_path, args.base_path, args.expected_count)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_token_match and result['token_mismatch_task_ids']:
        raise SystemExit('Token mismatches detected; investigate before full runs')


if __name__ == '__main__':
    main()
