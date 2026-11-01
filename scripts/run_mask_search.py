"""Prepare immutable SWIFT snapshots and run controlled mask-search ablations."""
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
from model.mask_search import POLICIES, PROTOCOL
from run_experiments import (export_source, code_sources, save_changes, check_environment,
                             run, note, score_records, compare_scores)
from source_patches import prepare, replace
from research_utils import (aggregate, compare_aggregates, read_records, validate_pair, write_json)

METHODS = ('baseline', 'swift', *POLICIES, 'adaptive', 'fixed_first')
FIELDS = ['metric_protocol', 'search_protocol', 'stage', 'split', 'seed', 'dataset', 'model',
          'method', 'samples', 'tokens', 'inference_seconds', 'tokens_per_second',
          'global_tokens_per_second', 'speedup_vs_baseline', 'speedup_vs_swift',
          'inference_time_speedup_vs_swift', 'mean_accepted_length', 'acceptance_rate',
          'token_mismatch_vs_baseline', 'token_mismatch_vs_swift', 'p50_request_seconds',
          'p95_request_seconds', 'cache_hit_rate', 'embedding_seconds',
          'candidate_input_tokens', 'mask_search_seconds', 'optimization_host_seconds',
          'diagnostic_probe_seconds', 'search_event_count', 'accuracy', 'correct',
          'missing_answer_rate', 'token_limit_rate', 'correct_to_wrong_vs_baseline',
          'wrong_to_correct_vs_baseline', 'pass_at_1', 'max_new_tokens']


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
                       '    args = parser.parse_args()')
        write_text(path, text)


def patch_research(root):
    path = root / 'evaluation_llama/inference_swift.py'
    text = path.read_text(encoding='utf-8')
    text = replace(text, '    args = parser.parse_args()', '''    parser.add_argument('--search-policy', choices=['fixed_first', 'match_uniform', 'cost_uniform', 'match_halving', 'cost_halving', 'cost_halving_refresh'], required=True)
    parser.add_argument('--search-token-budget', type=int, default=824)
    parser.add_argument('--search-interval', type=int, default=32)
    parser.add_argument('--search-time-budget-ms', type=float, default=0)
    parser.add_argument('--diagnostic-samples', type=int, default=0)
    args = parser.parse_args()''')
    text = replace(text, '    run_eval(', '''    search = None
    if args.search_policy != 'fixed_first':
        from model.mask_search import MaskSearch
        search = MaskSearch(model, args.search_policy, seed=args.seed,
                            token_budget=args.search_token_budget, interval=args.search_interval,
                            time_budget_ms=args.search_time_budget_ms,
                            diagnostic_samples=args.diagnostic_samples)
        swift_optimization = search.optimize
    fixed_first_seen = False

    run_eval(''')
    text = replace(text, '    begin_request(statistics)', '''    begin_request(statistics)
    global fixed_first_seen
    if search is not None:
        search.begin_request(swift_forward.request_context, statistics)
    elif fixed_first_seen:
        statistics['optimization'] = False''')
    text = replace(text, '        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist() or new_token_num >= max_new_tokens:',
                   '        if search is not None:\n'
                   '            search.observe_commit(accept_length_tree, len(top1_prob))\n'
                   '        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist() or new_token_num >= max_new_tokens:')
    text = replace(text, '                    utility=utility)',
                   '                    utility=utility,\n'
                   '                    **({"verified_past_key_values_data": past_key_values_data,\n'
                   '                        "verified_current_length_data": current_length_data} if search is not None else {}))')
    text = replace(text, '                and idx % statistics["opt_interval"] == 0):',
                   '                and idx % statistics["opt_interval"] == 0\n'
                   '                and (search is None or search.ready(new_token_num, statistics))):')
    text = replace(text, '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num',
                   '    if search is not None:\n'
                   '        swift_forward.last_metrics.update(search.metrics())\n'
                   '    fixed_first_seen = True\n'
                   '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num')
    write_text(path, text)
    shutil.copyfile(REPO / 'model/mask_search.py', root / 'model/mask_search.py')


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
        'mask_search_sha256': hashlib.sha256((REPO/'model/mask_search.py').read_bytes()).hexdigest(),
        'note': 'Only optimizer hook differs; original acceptance/model utility files are unchanged'})
    compile((research/'model/mask_search.py').read_text(encoding='utf-8'), 'model/mask_search.py', 'exec')
    return {'baseline': swift, 'swift': swift, 'adaptive': adaptive,
            **{p: research for p in (*POLICIES, 'fixed_first')}}, sources


def configurations(args):
    for index, model in enumerate(args.models):
        for task in args.tasks:
            if task == 'humaneval':
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


def write_diagnostics(folder, task, method, records):
    """Export candidate rankings vs paid linear probes and subsequent tree segments."""
    rows = []
    for record in records:
        for event_index,event in enumerate(record.get('forward_metrics',{}).get('search_events',[])):
            latest = {item['candidate']: item for item in event['observations']}
            for index,probe in enumerate(event['diagnostic']):
                observation = latest[index]
                rounds = event['future_rounds']
                rows.append({'task_id':record['task_id'],'method':method,'event':event_index,'candidate':index,
                             'checkpoint_tokens':event['checkpoint_tokens'],'context_offset':event['context_offset'],
                             'mask':json.dumps(probe['mask'],sort_keys=True),
                             'labels_evaluated':observation['labels'],'matchness':observation['matchness'],
                             'prefix_yield_proxy':observation['prefix_yield_proxy'],
                             'utility_proxy':observation['utility_proxy'],
                             'linear_committed_tokens':probe['linear_committed_tokens'],
                             'linear_tokens_per_second':probe['linear_tokens_per_second'],
                             'probe_seconds':probe['seconds_including_copy'],
                             'selected':probe['mask']==event['selected'],
                             'selected_future_tree_M':event['future_committed_tokens']/rounds if rounds else None,
                             'scope':'paid_linear_probe_not_tree_benchmark'})
    if rows:
        with open(folder/f'{task}.{method}.diagnostics.csv','x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(folder, task, actual_task, model, count, seed, args):
    records, summaries, metadata, manifests = {}, {}, [], []
    for method in args.methods:
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
    gsm_scores = {m: score_records(records[m], count) for m in args.methods} if actual_task == 'gsm8k' else {}
    for method in args.methods:
        for base in references:
            validate_pair(records[method], records[base])
        metrics = [r.get('forward_metrics', {}) for r in records[method]]
        write_diagnostics(folder,task,method,records[method])
        summary = summaries[method]
        row = {k: summary[k] for k in ('metric_protocol','samples','tokens','inference_seconds',
               'tokens_per_second','global_tokens_per_second','mean_accepted_length','acceptance_rate',
               'cache_hit_rate','embedding_seconds','optimization_host_seconds')}
        times = [sum(r['choices'][0]['wall_time']) for r in records[method]]
        row.update(search_protocol=PROTOCOL, stage=args.stage, split=args.split,
                   seed=seed, dataset=task, model=model, method=method,
                   max_new_tokens=metadata[0]['config']['max_new_tokens'],
                   p50_request_seconds=percentile(times,.5), p95_request_seconds=percentile(times,.95))
        for base in references:
            ratio = compare_aggregates(summary, summaries[base])
            row['speedup_vs_'+base] = ratio['speedup_ratio']
            if base == 'swift':
                row['inference_time_speedup_vs_swift'] = ratio['inference_time_speedup']
            row['token_mismatch_vs_'+base] = sum(a['output_token_ids'] != b['output_token_ids']
                                                for a,b in zip(records[method], records[base]))
        for key in ('candidate_input_tokens','mask_search_seconds','diagnostic_probe_seconds'):
            row[key] = sum(m.get(key,0) for m in metrics)
        row['search_event_count'] = sum(len(m.get('search_events',[])) for m in metrics)
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
    parser.add_argument('--tasks',nargs='+',choices=['cnndm','gsm8k','humaneval'],default=['cnndm','gsm8k'])
    parser.add_argument('--gsm8k-profiles',nargs='+',choices=['paper64','quality512'],default=['paper64','quality512'])
    parser.add_argument('--cnndm-max-new-tokens',nargs='+',type=int,choices=[64,512],default=[64])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[2024])
    parser.add_argument('--data-seed',type=int,default=2024)
    parser.add_argument('--data-num',type=int)
    parser.add_argument('--search-token-budget',type=int,default=824)
    parser.add_argument('--search-interval',type=int,default=32)
    parser.add_argument('--search-time-budget-ms',type=float,default=0)
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
    if (args.data_num < 1 or args.search_token_budget < 206 or args.search_interval < 1
            or not math.isfinite(args.search_time_budget_ms) or args.search_time_budget_ms < 0):
        parser.error('Invalid positive count/search budget/interval or finite time budget')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output_dir or REPO/'_local/runs'/f'mask_search_{stamp}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        roots, sources = prepare_sources(args,output)
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config.update(sources=sources,search_protocol=PROTOCOL,
                      note='diagnose includes paid counterfactual probes; not a performance benchmark')
        write_json(output/'run_config.json',config)
        shutil.copytree(SCRIPTS,output/'driver/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(REPO/'model/mask_search.py',output/'driver/mask_search.py')
        if args.prepare_only:
            write_json(output/'status.json',{'status':'prepared_only','inference_ran':False})
            note(f'PREPARED: {output}; no GPU inference.')
            return
        check_environment(output)
        rows = []
        for repetition,seed in enumerate(args.seeds):
            for task,actual_task,model,count,limit,top_p in configurations(args):
                model_id = model.rstrip('/').split('/')[-1]
                folder = output/f'seed_{seed}'/model_id
                folder.mkdir(parents=True,exist_ok=True)
                order = args.methods[repetition % len(args.methods):] + args.methods[:repetition % len(args.methods)]
                for method in order:
                    module = 'inference_baseline' if method == 'baseline' else 'inference_swift'
                    command = [sys.executable,'-m','evaluation_llama.'+module,'--model-path',model,
                               '--model-id',model_id,'--task-name',actual_task,'--data-num',count,
                               '--max-new-tokens',limit,'--temperature',0,'--top-p',top_p,
                               '--dtype','float16','--seed',seed,'--data-seed',args.data_seed,
                               '--research-split',args.split,'--answer-file',folder/f'{task}.{method}.jsonl']
                    if method != 'baseline':
                        command += ['--optimization','--bayes','--skip-ratio',.45]
                    if method == 'adaptive':
                        command += ['--cache-enabled','--cache-clear','--cache-threshold',.85,'--cache-max-size',100]
                    elif method in (*POLICIES,'fixed_first'):
                        command += ['--search-policy',method,'--search-token-budget',args.search_token_budget,
                                    '--search-interval',args.search_interval,'--search-time-budget-ms',args.search_time_budget_ms,
                                    '--diagnostic-samples',4 if args.stage == 'diagnose' else 0]
                    note(f'{args.stage}: {model_id}/{task}/{method}, seed {seed}, N={count}')
                    run(command,roots[method])
                rows.extend(summarize(folder,task,actual_task,model,count,seed,args))
                write_results(output,rows)
        write_json(output/'status.json',{'status':'complete','inference_ran':True,'stage':args.stage,'rows':len(rows)})
        note(f'DONE: {output/"results.csv"}')
        if args.stage == 'diagnose':
            note('Diagnose results include probes; do not report them as benchmark speedups.')
    except Exception as error:
        write_json(output/'failure.json',{'status':'failed','error':repr(error)})
        raise


if __name__ == '__main__':
    main()
