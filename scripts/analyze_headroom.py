"""Offline analysis for diagnostics D1/D2/D4. CPU only; reads run JSONL, never runs a model.

  bank     Collect request-end masks from earlier runs into a mask bank (input of D2).
  analyze  For a mask-search run directory:
           D2  oracle headroom: best single mask, per-domain and per-request oracles
               over fixed_mask runs vs SWIFT / fixed_first / AR, with a timing-noise
               estimate from a replicated mask (name ending in `_rep`).
           D4  proxy alignment: does a higher M / alpha pick the faster mask?
           D1  first-request lottery: spread of SWIFT speedup and end masks over
               stream-order seeds.

All speed figures follow swift_author_code_v1: mean over requests of tokens_i/time_i.
Per-request oracles select on the same noisy measurement they report, so they are
upper bounds; the replicate-based inflation must be subtracted before claiming headroom.
"""
import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
from research_utils import read_records, validate_pair  # noqa: E402

REFERENCES = ('baseline', 'swift', 'fixed_first', 'adaptive')


# ---------------------------------------------------------------- statistics
def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci(values, statistic=statistics.fmean, rounds=2000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(rounds))
    return draws[int(.025 * rounds)], draws[int(.975 * rounds) - 1]


# ---------------------------------------------------------------- record views
def request_view(record):
    choice = record['choices'][0]
    tokens, seconds = sum(choice['new_tokens']), sum(choice['wall_time'])
    lengths = choice.get('accept_lengths') or []
    return {'task_id': record['task_id'], 'domain': record.get('item_task') or record.get('task_name'),
            'speed': tokens / seconds, 'seconds': seconds, 'tokens': tokens,
            'M': sum(lengths) / len(lengths) if lengths else None,
            'alpha': choice.get('acceptance_rate'),
            'mask': used_mask(record)}


def used_mask(record):
    """Mask that served the request (AdaptiveCache hits restore the stream incumbent afterwards)."""
    metrics = record.get('forward_metrics', {})
    return canonical(metrics.get('used_layers', metrics.get('incumbent_layers')))


def canonical(mask):
    if not mask:
        return None
    return (tuple(sorted(map(int, mask[0]))), tuple(sorted(map(int, mask[1]))))


def load_folder(folder):
    """{task_key: {method_name: [request views]}} for one seed/order/model folder."""
    runs = defaultdict(dict)
    for path in sorted(folder.glob('*.jsonl')):
        if path.name.endswith('.samples.jsonl'):
            continue
        task, method = path.name[:-len('.jsonl')].rsplit('.', 1)
        summary = Path(str(path) + '.summary.json')
        if not summary.exists() or json.loads(summary.read_text(encoding='utf-8')).get('status') != 'complete':
            print(f'skip incomplete {path}', file=sys.stderr)
            continue
        runs[task][method] = read_records(path)
    return runs


def mean_speed(views):
    return statistics.fmean(v['speed'] for v in views)


# ---------------------------------------------------------------- D2 + D4
def headroom(task, methods):
    masks = {m[len('mask_'):]: methods[m] for m in methods if m.startswith('mask_')}
    if not masks:
        return None
    anchor = next(iter(methods.values()))
    for records in methods.values():
        validate_pair(records, anchor)
    views = {name: [request_view(r) for r in records] for name, records in masks.items()}
    refs = {name: [request_view(r) for r in methods[name]] for name in REFERENCES if name in methods}
    ar = mean_speed(refs['baseline']) if 'baseline' in refs else None
    n = len(anchor)
    names = [k for k in views if not k.endswith('_rep')]
    speed = {k: [v['speed'] for v in views[k]] for k in views}

    def ratio(value):
        return value / ar if ar else None

    out = {'task': task, 'requests': n, 'bank_size': len(names), 'ar_tokens_per_second': ar}
    for name, vs in refs.items():
        out[f'{name}_speedup'] = ratio(mean_speed(vs))
    per_mask = {k: statistics.fmean(speed[k]) for k in names}
    best = max(per_mask, key=per_mask.get)
    out.update(best_single_mask=best, best_single_speedup=ratio(per_mask[best]),
               worst_single_speedup=ratio(min(per_mask.values())),
               median_single_speedup=ratio(statistics.median(per_mask.values())))
    oracle = [max(speed[k][i] for k in names) for i in range(n)]
    out['oracle_request_speedup'] = ratio(statistics.fmean(oracle))
    domains = sorted({v['domain'] for v in views[names[0]]})
    if len(domains) > 1:
        total = 0.0
        for d in domains:
            idx = [i for i, v in enumerate(views[names[0]]) if v['domain'] == d]
            total += max(sum(speed[k][i] for i in idx) for k in names)
            out[f'best_mask_{d}'] = max(names, key=lambda k: sum(speed[k][i] for i in idx))
        out['oracle_domain_speedup'] = ratio(total / n)
    # Timing noise: the same mask run twice. Max of two identical runs is pure selection bias.
    reps = [k for k in views if k.endswith('_rep') and k[:-4] in views]
    if reps:
        k = reps[0][:-4]
        pair_max = statistics.fmean(max(a, b) for a, b in zip(speed[k], speed[k + '_rep']))
        pair_mean = statistics.fmean((a + b) / 2 for a, b in zip(speed[k], speed[k + '_rep']))
        out['noise_selection_inflation'] = pair_max / pair_mean - 1
        out['replicate_request_cv'] = statistics.fmean(
            abs(a - b) / ((a + b) / 2) for a, b in zip(speed[k], speed[k + '_rep'])) / math.sqrt(2)
        out['output_mismatch_replicate'] = sum(
            a['tokens'] != b['tokens'] for a, b in zip(views[k], views[k + '_rep']))
    if 'fixed_first' in refs:
        ff = mean_speed(refs['fixed_first'])
        out['oracle_request_gain_vs_fixed_first'] = statistics.fmean(oracle) / ff - 1
        out['best_single_gain_vs_fixed_first'] = per_mask[best] / ff - 1
    # D4: within each request, does the mask with higher M / alpha also run faster?
    for proxy in ('M', 'alpha'):
        rhos, regrets = [], []
        for i in range(n):
            xs = [views[k][i][proxy] for k in names]
            ys = [speed[k][i] for k in names]
            if any(x is None for x in xs):
                continue
            rho = spearman(xs, ys)
            if rho is not None:
                rhos.append(rho)
            pick = max(range(len(names)), key=lambda j: (xs[j], -j))
            regrets.append(1 - ys[pick] / max(ys))
        out[f'{proxy}_within_request_spearman'] = statistics.fmean(rhos) if rhos else None
        out[f'{proxy}_argmax_speed_regret'] = statistics.fmean(regrets) if regrets else None
        mask_proxy = [statistics.fmean(v[proxy] for v in views[k] if v[proxy] is not None) for k in names]
        out[f'{proxy}_across_mask_pearson'] = pearson(mask_proxy, [per_mask[k] for k in names])
    gains = [o / s - 1 for o, s in zip(oracle, speed[best])]
    out['oracle_vs_best_single_ci95'] = bootstrap_ci(gains)
    return out, {k: ratio(v) for k, v in per_mask.items()}


# ---------------------------------------------------------------- D1
def lottery(run_dir):
    rows = []
    for folder in sorted(run_dir.glob('seed_*/order_*/*')):
        for task, methods in load_folder(folder).items():
            if 'swift' not in methods or 'baseline' not in methods:
                continue
            swift = [request_view(r) for r in methods['swift']]
            ar = mean_speed([request_view(r) for r in methods['baseline']])
            row = {'order': folder.parent.name, 'model': folder.name, 'task': task,
                   'swift_speedup': mean_speed(swift) / ar,
                   'swift_M': statistics.fmean(v['M'] for v in swift if v['M'] is not None),
                   'swift_end_mask': json.dumps(swift[-1]['mask']),
                   'swift_distinct_masks': len({v['mask'] for v in swift})}
            if 'fixed_first' in methods:
                row['fixed_first_speedup'] = mean_speed([request_view(r) for r in methods['fixed_first']]) / ar
            rows.append(row)
    summary = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['task'])].append(row)
    for (model, task), items in sorted(groups.items()):
        values = [r['swift_speedup'] for r in items]
        summary.append({'model': model, 'task': task, 'orders': len(items),
                        'swift_speedup_mean': statistics.fmean(values),
                        'swift_speedup_min': min(values), 'swift_speedup_max': max(values),
                        'swift_speedup_cv': statistics.pstdev(values) / statistics.fmean(values),
                        'distinct_end_masks': len({r['swift_end_mask'] for r in items})})
    return rows, summary


# ---------------------------------------------------------------- bank
def uniform_mask(layers):
    odd = list(range(1, layers - 1, 2))
    return odd, odd


def random_mask(layers, skip_ratio, rng):
    population = 2 * (layers - 2)
    chosen = rng.sample(range(population), int(population * skip_ratio))
    return (sorted(i + 1 for i in chosen if i < layers - 2),
            sorted(i - (layers - 2) + 1 for i in chosen if i >= layers - 2))


def build_bank(args):
    seen, bank = {}, []

    def add(name, mask, source):
        key = canonical(mask)
        if key in seen:
            seen[key]['sources'].append(source)
            return
        entry = {'name': name, 'attention': list(key[0]), 'mlp': list(key[1]), 'sources': [source]}
        seen[key] = entry
        bank.append(entry)

    layers = args.layers
    for run in args.runs:
        for path in sorted(Path(run).rglob('*.jsonl')):
            if path.name.endswith('.samples.jsonl'):
                continue
            method = path.name[:-len('.jsonl')].rsplit('.', 1)[1]
            if method not in args.methods:
                continue
            records = read_records(path)
            meta = Path(str(path) + '.metadata.json')
            if meta.exists():
                found = json.loads(meta.read_text(encoding='utf-8'))['model_config']['num_hidden_layers']
                if layers and layers != found:
                    raise ValueError(f'{path}: {found} layers, bank expects {layers} (one model per bank)')
                layers = found
            masks = []
            for r in records:
                m = used_mask(r)
                if m and m not in masks:
                    masks.append(m)
            for m in masks:
                add(f'm{len(bank):02d}', m, str(path.relative_to(run)))
    if not layers:
        raise ValueError('Number of layers unknown; pass --layers')
    add(f'm{len(bank):02d}', uniform_mask(layers), 'uniform_init')
    rng = random.Random(args.seed)
    for _ in range(args.random):
        add(f'm{len(bank):02d}', random_mask(layers, args.skip_ratio, rng), f'random_seed{args.seed}')
    if args.replicate and bank:
        first = bank[0]
        bank.append(dict(first, name=first['name'] + '_rep', sources=['timing_noise_replicate']))
    out = {'layers': layers, 'skip_ratio_random': args.skip_ratio, 'masks': bank}
    Path(args.output).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(f'{len(bank)} masks -> {args.output}')


# ---------------------------------------------------------------- output
def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for row in rows for k in row}, key=lambda k: (k not in ('task', 'model', 'order'), k))
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    run_dir = Path(args.run).resolve()
    out_dir = Path(args.output or run_dir / 'analysis_headroom')
    out_dir.mkdir(parents=True, exist_ok=True)
    head, per_mask_rows = [], []
    for folder in sorted({p.parent for p in run_dir.glob('seed_*/*/*/*.jsonl')}):
        for task, methods in load_folder(folder).items():
            result = headroom(task, methods)
            if result is None:
                continue
            row, per_mask = result
            row.update(model=folder.name, order=folder.parent.name)
            row['oracle_vs_best_single_ci95'] = json.dumps(row['oracle_vs_best_single_ci95'])
            head.append(row)
            per_mask_rows += [{'model': folder.name, 'order': folder.parent.name, 'task': task,
                               'mask': k, 'speedup_vs_ar': v} for k, v in per_mask.items()]
    lot_rows, lot_summary = lottery(run_dir)
    write_csv(out_dir / 'd2_d4_headroom.csv', head)
    write_csv(out_dir / 'd2_per_mask.csv', per_mask_rows)
    write_csv(out_dir / 'd1_lottery_runs.csv', lot_rows)
    write_csv(out_dir / 'd1_lottery_summary.csv', lot_summary)
    keys = ('task', 'model', 'swift_speedup', 'fixed_first_speedup', 'best_single_speedup',
            'oracle_domain_speedup', 'oracle_request_speedup', 'noise_selection_inflation',
            'M_within_request_spearman', 'M_argmax_speed_regret')
    for row in head:
        print(' | '.join(f'{k}={row[k]:.4f}' if isinstance(row.get(k), float) else f'{k}={row.get(k)}'
                         for k in keys))
    for row in lot_summary:
        print('D1', row)
    print(f'-> {out_dir}')


def main():
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bank')
    b.add_argument('--runs', nargs='+', required=True, help='Run directories to harvest masks from')
    b.add_argument('--methods', nargs='+', default=['swift', 'fixed_first', 'adaptive'])
    b.add_argument('--layers', type=int, default=None)
    b.add_argument('--random', type=int, default=4, help='Extra random masks (lower-bound arms)')
    b.add_argument('--skip-ratio', type=float, default=.45)
    b.add_argument('--seed', type=int, default=2024)
    b.add_argument('--replicate', action='store_true', help='Duplicate first mask to measure timing noise')
    b.add_argument('--output', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--run', required=True)
    a.add_argument('--output')
    args = parser.parse_args()
    build_bank(args) if args.command == 'bank' else analyze(args)


if __name__ == '__main__':
    main()
