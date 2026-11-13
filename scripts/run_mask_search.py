"""Prepare immutable SWIFT snapshots and run SkipBook/PrefillCache experiments and ablations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path[:0] = [str(REPO), str(SCRIPTS / 'runtime')]
from run_experiments import (export_source, code_sources, save_changes, check_environment,
                             run, note, score_records, compare_scores)
from source_patches import prepare, replace
from research_utils import (aggregate, compare_aggregates, read_records, validate_pair, write_json)

PROTOCOL = 'skipbook_v1'
BOOK_METHODS = ('prefill_cache', 'skipbook')
# Named SkipBook ablations: one runner invocation, one results.csv, comparable rows.
ABLATIONS = {
    'skipbook_k1': {'sb_max_arms': 1},          # no bandit: one mask per context
    'skipbook_nodrift': {'sb_drift': 'off'},    # no Page-Hinkley re-exploration
    'skipbook_onectx': {'sb_threshold': -1.0},  # no prompt contexts: one global book
    'skipbook_q64': {'sb_key_pool': 'last64'},  # key from the last 64 prompt tokens only
}
RESEARCH_METHODS = ('fixed_first', 'fixed_mask', *BOOK_METHODS, *ABLATIONS)
METHODS = ('baseline', 'swift', 'adaptive', *RESEARCH_METHODS)
DEFAULT_METHODS = ('baseline', 'swift', 'fixed_first', 'prefill_cache', 'skipbook')
SB_KEYS = ('sb_threshold', 'sb_max_arms', 'sb_gamma', 'sb_ucb_c', 'sb_drift', 'sb_key_pool')
FIELDS = ['metric_protocol', 'search_protocol', 'stage', 'split', 'seed', 'dataset', 'model',
          'method', 'samples', 'tokens', 'inference_seconds', 'tokens_per_second',
          'global_tokens_per_second', 'speedup_vs_baseline', 'speedup_vs_swift',
          'inference_time_speedup_vs_swift', 'mean_accepted_length', 'acceptance_rate',
          'token_mismatch_vs_baseline', 'token_mismatch_vs_swift', 'p50_request_seconds',
          'p95_request_seconds', 'cache_hit_rate', 'embedding_seconds', 'optimization_host_seconds',
          'skipbook_explore_share', 'skipbook_contexts', 'skipbook_drift_events', 'accuracy', 'correct',
          'missing_answer_rate', 'token_limit_rate', 'correct_to_wrong_vs_baseline',
          'wrong_to_correct_vs_baseline', 'pass_at_1', 'max_new_tokens',
          'stream_order_seed', 'mix_ratio', 'fixed_mask']


def write_text(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def patch_split(root):
    """Research snapshots only: hold out train examples from few-shot demonstrations."""
    path = root / 'evaluation_llama/eval.py'
    text = path.read_text(encoding='utf-8')
    text = replace(text, 'def load_data(task_name, seed, data_num=10):',
                   "def load_data(task_name, seed, data_num=10, split='test'):")
    text = replace(text, '        data, shots, metadata = load_gsm8k(seed, data_num)',
                   '        data, shots, metadata = load_gsm8k(seed, data_num, split=split)')
    text = replace(text, "dataset = load_dataset('cnn_dailymail', name='3.0.0', split='test')",
                   "dataset = load_dataset('cnn_dailymail', name='3.0.0', split=split)")
    text = replace(text, '        data = list(dataset.shuffle(seed=seed).select(range(data_num)))',
                   '        offset = 1 if split == "train" else 0\n'
                   '        if data_num + offset > len(dataset):\n'
                   '            raise ValueError("Insufficient train questions after excluding the demonstration")\n'
                   '        data = list(dataset.shuffle(seed=seed).select(range(offset, offset+data_num)))')
    text = replace(text, "    if task_name == 'humaneval':\n        from human_eval.data",
                   "    if task_name == 'humaneval':\n        if split != 'test':\n"
                   "            raise ValueError('HumanEval is not a development/tuning set')\n        from human_eval.data")
    text = replace(text, '        data, shots = load_data(task_name, data_seed, data_num)',
                   '        research_split = (run_config or {}).get("research_split", "test")\n'
                   '        data, shots = load_data(task_name, data_seed, data_num, split=research_split)')
    text = replace(text, "        manifest = {'task_name': task_name, 'data_seed': data_seed,",
                   "        manifest = {'task_name': task_name, 'data_seed': data_seed, 'split': research_split,")
    text = text.replace("'cnn_dailymail/3.0.0/test'", "f'cnn_dailymail/3.0.0/{research_split}'")
    text = text.replace("'openai/gsm8k/main/test'", "f'openai/gsm8k/main/{research_split}'")
    write_text(path, text)
    path = root / 'evaluation_llama/gsm8k.py'
    text = path.read_text(encoding='utf-8')
    text = replace(text, 'def load_gsm8k(seed, count):', "def load_gsm8k(seed, count, split='test'):")
    text = replace(text, "'main', split='test')", "'main', split=split)")
    text = replace(text, "f'gsm8k/test/{i}'", "f'gsm8k/{split}/{i}'")
    text = replace(text, '    data = list(test.shuffle(seed=seed).select(range(count)))',
                   '    offset = 5 if split == "train" else 0\n'
                   '    if count + offset > len(test):\n'
                   '        raise ValueError("Insufficient train questions after excluding five demonstrations")\n'
                   '    data = list(test.shuffle(seed=seed).select(range(offset, offset+count)))')
    text = replace(text, "'dataset': 'openai/gsm8k/main/test'", "'dataset': f'openai/gsm8k/main/{split}'")
    write_text(path, text)
    # Both baseline and SWIFT accept the split, but retain their decoder implementation.
    for name in ('inference_baseline.py', 'inference_swift.py'):
        path = root / 'evaluation_llama' / name
        text = path.read_text(encoding='utf-8')
        text = replace(text, '    args = parser.parse_args()',
                       "    parser.add_argument('--research-split', choices=['train', 'test'], default='test')\n"
                       "    parser.add_argument('--stream-order-seed', type=int, default=None)\n"
                       "    parser.add_argument('--mix-ratio', type=float, default=None)\n"
                       '    args = parser.parse_args()')
        write_text(path, text)


def patch_research(root):
    path = root / 'evaluation_llama/inference_swift.py'
    text = path.read_text(encoding='utf-8')
    text = replace(text, '    args = parser.parse_args()', '''    parser.add_argument('--fixed-mask', default=None, help='JSON attention/mlp lists, frozen for the whole stream')
    parser.add_argument('--sb-threshold', type=float, default=0.85)
    parser.add_argument('--sb-max-arms', type=int, default=3)
    parser.add_argument('--sb-capacity', type=int, default=32)
    parser.add_argument('--sb-gamma', type=float, default=0.95)
    parser.add_argument('--sb-ucb-c', type=float, default=0.1)
    parser.add_argument('--sb-min-pulls', type=int, default=3)
    parser.add_argument('--sb-drift', choices=['on', 'off'], default='on')
    parser.add_argument('--sb-key-pool', default='mean')
    parser.add_argument('--search-policy', choices=['fixed_first', 'fixed_mask', 'prefill_cache', 'skipbook'], required=True)
    args = parser.parse_args()''')
    text = replace(text, '    run_eval(', '''    frozen_stream = args.search_policy == 'fixed_mask'
    if frozen_stream:
        import json as _json
        _mask = _json.loads(args.fixed_mask)
        model.set_skip_layers(np.array(_mask['attention'], dtype=int), np.array(_mask['mlp'], dtype=int))
        statistics['optimization'] = False
    elif args.fixed_mask is not None:
        raise ValueError('--fixed-mask requires --search-policy fixed_mask')
    book = None
    forward_entry = swift_forward
    if args.search_policy in ('prefill_cache', 'skipbook'):
        from model.skipbook import SkipBook

        def make_optimizer():
            # Same construction as SWIFT's stream optimizer above.
            fresh = BayesianOptimization(f=None, pbounds=pbounds, random_state=1, verbose=1,
                                         allow_duplicate_points=True)
            fresh.set_gp_params(alpha=1e-2)
            return fresh
        book = SkipBook(model, statistics, make_optimizer, variant=args.search_policy,
                        threshold=args.sb_threshold, max_arms=args.sb_max_arms, capacity=args.sb_capacity,
                        gamma=args.sb_gamma, ucb_c=args.sb_ucb_c, min_pulls=args.sb_min_pulls,
                        drift=args.sb_drift == 'on', key_pool=args.sb_key_pool, stream_optimizer=optimizer)
        initialize_swift = book.hook_initialize(swift_verify, swift_draft)
        optimizer = book.optimizer
        forward_entry = book.wrap(swift_forward)
    fixed_first_seen = False

    run_eval(''')
    text = replace(text, '    begin_request(statistics)', '''    begin_request(statistics)
    global fixed_first_seen
    if (fixed_first_seen and args.search_policy == 'fixed_first') or frozen_stream:
        statistics['optimization'] = False''')
    text = replace(text, '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num',
                   '    fixed_first_seen = True\n'
                   '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num')
    text = replace(text, '        forward_func=swift_forward,', '        forward_func=forward_entry,')
    write_text(path, text)
    shutil.copyfile(REPO / 'model/skipbook.py', root / 'model/skipbook.py')


def prepare_sources(args, output):
    snapshots = output / 'snapshots'
    snapshots.mkdir()
    swift, adaptive = output / 'sources/swift', output / 'sources/adaptive'
    swift.mkdir(parents=True)
    adaptive.mkdir(parents=True)
    sources = {'swift': export_source(args.swift_ref, swift, 'swift', snapshots),
               'adaptive': export_source(args.adaptive_ref, adaptive, 'adaptive', snapshots)}
    originals = {str(p): code_sources(p) for p in (swift, adaptive)}
    prepare(swift, adaptive, SCRIPTS / 'runtime')
    for root in (swift, adaptive):
        patch_split(root)
        save_changes(root, originals[str(root)])
    research = output / 'sources/research'
    shutil.copytree(swift, research)
    patch_research(research)
    save_changes(research, originals[str(swift)])
    write_json(research / 'RESEARCH_SOURCE.json', {
        'protocol': PROTOCOL,
        'skipbook_sha256': hashlib.sha256((REPO/'model/skipbook.py').read_bytes()).hexdigest(),
        'note': 'Only mask selection hooks differ; original acceptance/model utility files are unchanged'})
    compile((research/'model/skipbook.py').read_text(encoding='utf-8'), 'model/skipbook.py', 'exec')
    return {'baseline': swift, 'swift': swift, 'adaptive': adaptive,
            **{p: research for p in RESEARCH_METHODS}}, sources


def method_instances(args):
    """(name, policy, entry): fixed_mask expands per bank entry; ablations map to skipbook + overrides."""
    instances = []
    for method in args.methods:
        if method == 'fixed_mask':
            instances += [('mask_'+entry['name'], method, entry) for entry in args.bank]
        elif method in ABLATIONS:
            instances.append((method, 'skipbook', {'sb': ABLATIONS[method]}))
        else:
            instances.append((method, method, None))
    return instances


def book_arguments(args, entry):
    """--sb-* flags for a book method: runner defaults, overridden by a named ablation."""
    values = {key: getattr(args, key) for key in SB_KEYS}
    values.update((entry or {}).get('sb', {}))
    command = []
    for key in SB_KEYS:
        command += ['--'+key.replace('_', '-'), values[key]]
    return command


def load_bank(path):
    bank = json.loads(Path(path).read_text(encoding='utf-8'))
    bank = bank['masks'] if isinstance(bank, dict) else bank
    names = [str(entry['name']) for entry in bank]
    if not bank or len(names) != len(set(names)):
        raise ValueError('Mask bank must be a nonempty list with unique names')
    for entry in bank:
        if not str(entry['name']).replace('_', '').replace('-', '').isalnum():
            raise ValueError(f'Mask name must be alphanumeric/_/-: {entry["name"]!r}')
        entry['attention'] = sorted(map(int, entry['attention']))
        entry['mlp'] = sorted(map(int, entry['mlp']))
    return bank


def configurations(args):
    for index, model in enumerate(args.models):
        for task in args.tasks:
            if task == 'mixed':
                for ratio in args.mix_ratios:
                    yield f'mixed.r{ratio:g}', task, model, args.data_num, args.mixed_max_new_tokens, .85
            elif task == 'humaneval':
                yield task, task, args.code_models[index], 164, 512, .95
            elif task == 'cnndm':
                for limit in args.cnndm_max_new_tokens:
                    key = task if limit == 64 else 'cnndm.quality512'
                    yield key, task, model, args.data_num, limit, .85
            else:
                for profile in args.gsm8k_profiles:
                    yield 'gsm8k.'+profile, task, model, args.data_num, {'paper64':64, 'quality512':512}[profile], .85


def percentile(values, probability):
    values = sorted(values)
    x = (len(values)-1)*probability
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def summarize(folder, task, actual_task, model, count, seed, args, order_seed=None):
    records, summaries, metadata, manifests = {}, {}, [], []
    instances = method_instances(args)
    names = [name for name, _, _ in instances]
    for method in names:
        path = folder / f'{task}.{method}.jsonl'
        records[method] = read_records(path)
        sidecar = json.loads(Path(str(path)+'.summary.json').read_text(encoding='utf-8'))
        if len(records[method]) != count or sidecar.get('status') != 'complete':
            raise ValueError(f'Incomplete inference: {path}')
        summaries[method] = aggregate(records[method])
        metadata.append(json.loads(Path(str(path)+'.metadata.json').read_text(encoding='utf-8')))
        manifests.append(json.loads(Path(str(path)+'.manifest.json').read_text(encoding='utf-8')))
    if any(m != manifests[0] for m in manifests):
        raise ValueError('Different dataset IDs, split, prompt or demonstrations between methods')
    for m in metadata[1:]:
        if m['model_revision'] != metadata[0]['model_revision']:
            raise ValueError('Different model revisions between methods')
        for key in ('model_path', 'dtype', 'temperature', 'top_p', 'seed', 'data_seed', 'max_new_tokens'):
            if m['config'][key] != metadata[0]['config'][key]:
                raise ValueError(f'Different method configuration: {key}')
    rows = []
    references = [m for m in ('baseline', 'swift') if m in records]
    gsm_scores = {m: score_records(records[m], count) for m in names} if actual_task == 'gsm8k' else {}
    for method, _, entry in instances:
        for base in references:
            validate_pair(records[method], records[base])
        metrics = [r.get('forward_metrics', {}) for r in records[method]]
        summary = summaries[method]
        row = {k: summary[k] for k in ('metric_protocol','samples','tokens','inference_seconds',
               'tokens_per_second','global_tokens_per_second','mean_accepted_length','acceptance_rate',
               'cache_hit_rate','embedding_seconds','optimization_host_seconds')}
        times = [sum(r['choices'][0]['wall_time']) for r in records[method]]
        row.update(search_protocol=PROTOCOL, stage=args.stage, split=args.split,
                   seed=seed, dataset=task, model=model, method=method,
                   max_new_tokens=metadata[0]['config']['max_new_tokens'],
                   stream_order_seed=order_seed, mix_ratio=metadata[0]['config'].get('mix_ratio'),
                   fixed_mask=json.dumps({'attention': entry['attention'], 'mlp': entry['mlp']}) if entry and 'attention' in entry else None,
                   p50_request_seconds=percentile(times,.5), p95_request_seconds=percentile(times,.95))
        for base in references:
            ratio = compare_aggregates(summary, summaries[base])
            row['speedup_vs_'+base] = ratio['speedup_ratio']
            if base == 'swift':
                row['inference_time_speedup_vs_swift'] = ratio['inference_time_speedup']
            row['token_mismatch_vs_'+base] = sum(a['output_token_ids'] != b['output_token_ids']
                                                for a,b in zip(records[method], records[base]))
        if any('skipbook_mode' in m for m in metrics):
            row['skipbook_explore_share'] = sum(m.get('skipbook_mode') == 'explore' for m in metrics)/len(metrics)
            row['skipbook_contexts'] = len({m.get('skipbook_context') for m in metrics})
            row['skipbook_drift_events'] = sum(bool(m.get('skipbook_drift')) for m in metrics)
        if method == 'baseline':
            row['acceptance_rate'] = None
        if gsm_scores:
            score = gsm_scores[method]
            paired = compare_scores(score, gsm_scores['baseline']) if 'baseline' in gsm_scores else None
            score_output = dict(score)
            if paired is not None:
                score_output['vs_baseline'] = paired
            write_json(folder / f'{task}.{method}.accuracy.json', score_output)
            row.update(accuracy=score['accuracy'],correct=score['correct'],
                       missing_answer_rate=score['missing_answer_rate'],token_limit_rate=score['token_limit_rate'])
            if paired is not None:
                row.update(correct_to_wrong_vs_baseline=len(paired['correct_to_wrong_ids']),
                           wrong_to_correct_vs_baseline=len(paired['wrong_to_correct_ids']))
        elif actual_task == 'humaneval':
            score_file = folder / f'{task}.{method}.pass1.json'
            run([sys.executable, SCRIPTS/'evaluate_humaneval.py', '--input', folder/f'{task}.{method}.jsonl',
                 '--samples', folder/f'{task}.{method}.samples.jsonl', '--score', score_file], REPO)
            row['pass_at_1'] = json.loads(score_file.read_text(encoding='utf-8'))['pass@1']
        rows.append(row)
    return rows


def write_results(output, rows):
    temporary = output / 'results.csv.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output/'results.csv')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage',choices=['diagnose','evaluate'],default='diagnose')
    parser.add_argument('--models',nargs='+',default=['meta-llama/Llama-2-7b-hf','meta-llama/Llama-2-13b-hf'])
    parser.add_argument('--code-models',nargs='+',default=['codellama/CodeLlama-7b-hf','codellama/CodeLlama-13b-hf'])
    parser.add_argument('--tasks',nargs='+',choices=['cnndm','gsm8k','humaneval','mixed'],default=['cnndm','gsm8k'])
    parser.add_argument('--mix-ratios',nargs='+',type=float,default=[0.0,0.3,0.7,1.0],
                        help='D3 mixed CNN/DM+GSM8K stream; 0 = two contiguous blocks')
    parser.add_argument('--mixed-max-new-tokens',type=int,choices=[64,512],default=64)
    parser.add_argument('--stream-order-seeds',nargs='+',type=int,default=None,
                        help='D1: permute request order per seed (default: loader order)')
    parser.add_argument('--sb-threshold',type=float,default=0.85,help='SkipBook/PrefillCache cosine threshold')
    parser.add_argument('--sb-max-arms',type=int,default=3,help='SkipBook arms per context (1 = no bandit)')
    parser.add_argument('--sb-gamma',type=float,default=0.95,help='Discounted-UCB discount')
    parser.add_argument('--sb-ucb-c',type=float,default=0.1,help='UCB width on mean-normalised rewards')
    parser.add_argument('--sb-drift',choices=['on','off'],default='on',help='Page-Hinkley re-exploration')
    parser.add_argument('--sb-key-pool',default='mean',help='mean or lastN (e.g. last64) prefill tokens')
    parser.add_argument('--mask-bank',type=Path,help='D2: JSON list of {name, attention, mlp} for fixed_mask')
    parser.add_argument('--gsm8k-profiles',nargs='+',choices=['paper64','quality512'],default=['paper64','quality512'])
    parser.add_argument('--cnndm-max-new-tokens',nargs='+',type=int,choices=[64,512],default=[64])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(DEFAULT_METHODS),
                        help='Ablations: ' + ', '.join(f'{k} {v}' for k, v in ABLATIONS.items()))
    parser.add_argument('--seeds',nargs='+',type=int,default=[2024])
    parser.add_argument('--data-seed',type=int,default=2024)
    parser.add_argument('--data-num',type=int)
    parser.add_argument('--swift-ref',default='main')
    parser.add_argument('--adaptive-ref',default='origin/AdaptiveCache')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    args.split = 'train' if args.stage == 'diagnose' else 'test'
    args.data_num = args.data_num if args.data_num is not None else (16 if args.stage == 'diagnose' else 1000)
    for values in (args.models,args.code_models,args.tasks,args.methods,args.seeds,args.gsm8k_profiles,args.cnndm_max_new_tokens):
        if len(set(values)) != len(values):
            parser.error('Duplicate models/tasks/methods/seeds/profiles are not supported')
    if args.split == 'train' and 'humaneval' in args.tasks:
        parser.error('HumanEval cannot be used for development; use --stage evaluate')
    if 'humaneval' in args.tasks and len(args.models) != len(args.code_models):
        parser.error('--models and --code-models must have equal length for HumanEval')
    args.stream_order_seeds = args.stream_order_seeds or [None]
    if len(set(args.stream_order_seeds)) != len(args.stream_order_seeds) or len(set(args.mix_ratios)) != len(args.mix_ratios):
        parser.error('Duplicate stream order seeds or mix ratios')
    if any(not 0 <= r <= 1 for r in args.mix_ratios):
        parser.error('Mix ratios must lie in [0, 1]')
    if ('fixed_mask' in args.methods) != (args.mask_bank is not None):
        parser.error('--mask-bank is required with, and only with, --methods fixed_mask')
    args.bank = load_bank(args.mask_bank) if args.mask_bank else []
    if args.data_num < 1:
        parser.error('--data-num must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output_dir or REPO/'_local/runs'/f'skipbook_{stamp}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        roots, sources = prepare_sources(args,output)
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config.update(sources=sources,search_protocol=PROTOCOL,
                      note='diagnose uses train splits for development; report evaluate (test) numbers')
        write_json(output/'run_config.json',config)
        shutil.copytree(SCRIPTS,output/'driver/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(REPO/'model/skipbook.py',output/'driver/skipbook.py')
        if args.prepare_only:
            write_json(output/'status.json',{'status':'prepared_only','inference_ran':False})
            note(f'PREPARED: {output}; no GPU inference.')
            return
        check_environment(output)
        rows = []
        instances = method_instances(args)
        repetition = 0
        for seed in args.seeds:
          for order_seed in args.stream_order_seeds:
            for task,actual_task,model,count,limit,top_p in configurations(args):
                model_id = model.rstrip('/').split('/')[-1]
                folder = output/f'seed_{seed}'/('loader_order' if order_seed is None else f'order_{order_seed}')/model_id
                folder.mkdir(parents=True,exist_ok=True)
                # Rotate execution order across repetitions to spread drift/thermal effects.
                shift = repetition % len(instances)
                repetition += 1
                for name,method,entry in instances[shift:] + instances[:shift]:
                    module = 'inference_baseline' if method == 'baseline' else 'inference_swift'
                    command = [sys.executable,'-m','evaluation_llama.'+module,'--model-path',model,
                               '--model-id',model_id,'--task-name',actual_task,'--data-num',count,
                               '--max-new-tokens',limit,'--temperature',0,'--top-p',top_p,
                               '--dtype','float16','--seed',seed,'--data-seed',args.data_seed,
                               '--research-split',args.split,'--answer-file',folder/f'{task}.{name}.jsonl']
                    if order_seed is not None:
                        command += ['--stream-order-seed',order_seed]
                    if actual_task == 'mixed':
                        command += ['--mix-ratio',float(task.split('.r',1)[1])]
                    if method != 'baseline':
                        command += ['--optimization','--bayes','--skip-ratio',.45]
                    if method == 'adaptive':
                        command += ['--cache-enabled','--cache-clear','--cache-threshold',.85,'--cache-max-size',100]
                    elif method in RESEARCH_METHODS:
                        command += ['--search-policy',method]
                    if method in BOOK_METHODS:
                        command += book_arguments(args,entry)
                    if method == 'fixed_mask':
                        command += ['--fixed-mask',json.dumps({'attention':entry['attention'],'mlp':entry['mlp']})]
                    note(f'{args.stage}: {model_id}/{task}/{name}, seed {seed}, order {order_seed}, N={count}')
                    run(command,roots[method])
                rows.extend(summarize(folder,task,actual_task,model,count,seed,args,order_seed))
                write_results(output,rows)
        write_json(output/'status.json',{'status':'complete','inference_ran':True,'stage':args.stage,'rows':len(rows)})
        note(f'DONE: {output/"results.csv"}')
        if args.stage == 'diagnose':
            note('Diagnose runs use train splits; report --stage evaluate numbers in the paper.')
    except Exception as error:
        write_json(output/'failure.json',{'status':'failed','error':repr(error)})
        raise


if __name__ == '__main__':
    main()
