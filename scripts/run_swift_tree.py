"""Isolated, greedy-only SWIFT tree-control experiments on one H100.

Original Git source is exported, normalized with the common harness, then copied
to research snapshots. No AdaptiveCache or mask-search modules are imported.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path[:0] = [str(REPO), str(SCRIPTS / 'assets')]
from model.swift_tree_control import PROTOCOL
from source_patches import prepare_single, replace
from run_experiments import export_source, code_sources, save_changes, run, note, check_environment
from research_utils import aggregate, compare_aggregates, validate_pair, read_records, write_json
from gsm8k import score_records, compare_scores

METHODS = ('baseline', 'swift', 'native_control', 'raw_dp', 'rank_dp', 'stop_only', 'joint')


def write_text(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def patch_split(root):
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
                   '            raise ValueError("Not enough questions excluding the train demonstration")\n'
                   '        data = list(dataset.shuffle(seed=seed).select(range(offset, offset+data_num)))')
    text = replace(text, "    if task_name == 'humaneval':\n        from human_eval.data",
                   "    if task_name == 'humaneval':\n        if split != 'test':\n"
                   "            raise ValueError('HumanEval is evaluation-only')\n        from human_eval.data")
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
                   '        raise ValueError("Not enough questions excluding five train demonstrations")\n'
                   '    data = list(test.shuffle(seed=seed).select(range(offset, offset+count)))')
    text = replace(text, "'dataset': 'openai/gsm8k/main/test'", "'dataset': f'openai/gsm8k/main/{split}'")
    write_text(path, text)
    for name in ('inference_baseline.py', 'inference_swift.py'):
        path = root / 'evaluation_llama' / name
        text = replace(path.read_text(encoding='utf-8'), '    args = parser.parse_args()',
                       "    parser.add_argument('--research-split', choices=['train', 'test'], default='test')\n"
                       '    args = parser.parse_args()')
        write_text(path, text)


def patch_control(root):
    """Hook the native loop. Original posterior/commit/model code stays byte-identical."""
    path = root / 'model/swift/utils.py'
    text = path.read_text(encoding='utf-8')
    text = replace(text, 'TOPK = 10  # topk for sparse tree',
                   'TOPK = 10  # topk for sparse tree\n_tree_control = None')
    start = text.index('@torch.no_grad()\ndef swift_draft(')
    end = text.index('\ndef reset_swift_mode(', start)
    block = text[start:end]
    block = replace(block, '    draft_past_key_values_data = []', '''    control = _tree_control
    if control is not None:
        control.begin_draft(model, int(current_length_data[0]),
                            min(max_step_draft, max(0, max_new_tokens-new_token_num-1)))
        if control.max_depth == 0:
            empty = torch.empty((0, 1, TOPK), device=input_ids.device)
            return (empty.long(), empty, []), []
    draft_past_key_values_data = []''')
    # Keep native forward/lm_head, just wrap their existing block in a phase timer.
    first = block.index('            with model.self_draft():')
    last = block.index('            if logits_processor is not None:', first)
    forward = block[first:last]
    wrapped = '            with control.phase("draft"):\n' + '\n'.join('    '+line for line in forward.rstrip().splitlines()) + '\n'
    block = block[:first] + wrapped + block[last:]
    block = replace(block, '            if current_threshold < stop_threshold or new_token_num + step_draft + 2 >= max_new_tokens:',
                    '''            keep_stopping = control.observe_draft(
                torch.gather(origin_draft_probs, -1, topk_index).reshape(-1).float().cpu().numpy(),
                topk_index.reshape(-1).tolist())
            if keep_stopping or new_token_num + step_draft + 2 >= max_new_tokens:''')
    first = block.index('    draft_past_key_values_data = []')
    last = block.index('    ss_token, ss_prob', first)
    cache_block = block[first:last]
    block = block[:first] + '    with control.phase("draft_kv"):\n' + '\n'.join('    '+line for line in cache_block.rstrip().splitlines()) + '\n\n' + block[last:]
    text = text[:start] + block + text[end:]
    write_text(path, text)
    path = root / 'evaluation_llama/inference_swift.py'
    text = path.read_text(encoding='utf-8')
    text = replace(text, 'import argparse', 'import argparse\nimport time')
    text = replace(text, '    begin_request(statistics)',
                   '    begin_request(statistics)\n    tree_control.begin_request()')
    text = replace(text, '        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")',
                   '''        round_start = tree_control.round_start
        native_choices = get_choices_list(top1_prob, logits_processor=logits_processor)
        native_widths = [sum(len(p) == d+1 for p in native_choices) for d in range(len(top1_prob))]
        swift_choices = tree_control.select(native_widths)''')
    text = replace(text, '        swift_buffers = generate_swift_buffers(swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device)',
                   '''        buffer_start = time.perf_counter()
        swift_buffers = tree_control.buffers(swift_choices, generate_swift_buffers,
                          model.model.layers[-1].self_attn.q_proj.weight.device)
        tree_control.cost.host = (tree_control.cost.host + [time.perf_counter()-buffer_start])[-64:]''')
    start = text.index('        logits, outputs = tree_decoding(')
    end = text.index('\n\n', start)
    block = text[start:end]
    text = text[:start] + '        with tree_control.phase("verify", int(tree_candidates.numel()), input_ids.shape[1]):\n' + '\n'.join('    '+line for line in block.splitlines()) + text[end:]
    for phase, marker in (('candidate','        candidates, cart_candidates_prob, tree_candidates = generate_candidates('),
                          ('posterior','        best_candidate, accept_length, sample_p = evaluate_posterior('),
                          ('commit','        input_ids, new_token_num, sample_token = update_inference_inputs(')):
        start = text.index(marker)
        end = text.index('\n\n', start)
        block = text[start:end]
        text = text[:start] + f'        with tree_control.phase("{phase}"):\n' + '\n'.join('    '+line for line in block.splitlines()) + text[end:]
    # Original return 'outputs' contains the un-retrieved model outputs; obtain
    # dense node logits using the EXISTING retrieved logits (no second lm_head).
    text = replace(text, '        accept_length_tree = input_ids.shape[1] - cur_length',
                   '''        # Read existing retrieved logits; no extra target forward/lm_head or dense copy.
        tree_control.feedback(logits, swift_choices, input_ids.shape[1]-cur_length,
                              len(top1_prob), time.perf_counter()-round_start,
                              eos=prefix[committed-1] == tokenizer.eos_token_id)
        accept_length_tree = input_ids.shape[1] - cur_length''')
    text = replace(text, 'total_acc_num / draft_token_num',
                   'total_acc_num / draft_token_num if draft_token_num else 0.0')
    text = replace(text, '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num',
                   '    swift_forward.last_metrics.update(tree_control.finish())\n'
                   '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num')
    text = replace(text, '    args = parser.parse_args()', '''    parser.add_argument('--tree-policy', choices=['native','raw_dp','rank_dp','stop_only','joint','fixed'], required=True)
    parser.add_argument('--tree-max-nodes', type=int, default=251)
    parser.add_argument('--tree-lookahead', type=int, default=2)
    parser.add_argument('--tree-min-labels', type=int, default=32)
    parser.add_argument('--fixed-depth', type=int, default=4)
    parser.add_argument('--fixed-width', type=int, default=3)
    args = parser.parse_args()
    if args.temperature != 0:
        parser.error('Tree-control v1 supports greedy decoding only')''')
    text = replace(text, '    run_eval(', '''    from model.swift_tree_control import TreeControl
    import model.swift.utils as swift_utils
    tree_control = TreeControl(args.tree_policy, args.tree_max_nodes, args.tree_lookahead,
                               args.fixed_depth, args.fixed_width, args.tree_min_labels)
    swift_utils._tree_control = tree_control
    run_eval(''')
    write_text(path, text)
    shutil.copyfile(REPO/'model/swift_tree_control.py', root/'model/swift_tree_control.py')


def prepare_sources(args, output):
    snapshots = output/'snapshots'
    snapshots.mkdir()
    native = output/'sources/swift'
    native.mkdir(parents=True)
    info = export_source(args.swift_ref, native, 'swift', snapshots)
    original = code_sources(native)
    prepare_single(native, SCRIPTS/'assets')
    patch_split(native)
    save_changes(native, original)
    controlled = output/'sources/controlled'
    shutil.copytree(native, controlled)
    patch_control(controlled)
    save_changes(controlled, original)
    write_json(controlled/'TREE_CONTROL_SOURCE.json', {
        'protocol': PROTOCOL, 'module_sha256': hashlib.sha256((REPO/'model/swift_tree_control.py').read_bytes()).hexdigest(),
        'scope': 'greedy comb scoring and draft stopping; native layer search/posterior/commit unchanged'})
    return native, controlled, info


def configurations(args):
    for index, model in enumerate(args.models):
        for task in args.tasks:
            if task == 'humaneval':
                yield task, task, args.code_models[index], 164, 512, .95
            elif task == 'cnndm':
                yield task, task, model, args.data_num, 64, .85
            else:
                for profile in args.gsm8k_profiles:
                    yield 'gsm8k.'+profile, task, model, args.data_num, {'paper64':64, 'quality512':512}[profile], .85


def percentile(values, p):
    return float(__import__('numpy').percentile(values, p))


def summarize(folder, task, actual_task, model, count, seed, args, methods):
    records, summaries, metas, manifests = {}, {}, [], []
    for method in methods:
        path = folder/f'{task}.{method}.jsonl'
        records[method] = read_records(path)
        summary = json.loads(Path(str(path)+'.summary.json').read_text(encoding='utf-8'))
        if len(records[method]) != count or summary.get('status') != 'complete':
            raise ValueError(f'Incomplete inference: {path}')
        summaries[method] = aggregate(records[method])
        metas.append(json.loads(Path(str(path)+'.metadata.json').read_text(encoding='utf-8')))
        manifests.append(json.loads(Path(str(path)+'.manifest.json').read_text(encoding='utf-8')))
    if any(m != manifests[0] for m in manifests):
        raise ValueError('Different prompts/dataset IDs/splits between methods')
    for meta in metas[1:]:
        if meta['model_revision'] != metas[0]['model_revision']:
            raise ValueError('Different model revisions')
        for key in ('model_path','dtype','seed','data_seed','temperature','top_p','max_new_tokens'):
            if meta['config'][key] != metas[0]['config'][key]:
                raise ValueError('Different run configuration: '+key)
    scores = {m:score_records(records[m], count) for m in methods} if actual_task == 'gsm8k' else {}
    rows = []
    for method in methods:
        base, swift = [compare_aggregates(summaries[method], summaries[m]) for m in ('baseline','swift')]
        for reference in ('baseline','swift'):
            validate_pair(records[method],records[reference])
        if any('output_token_ids' not in r for m in methods for r in records[m]):
            raise ValueError('Token mismatch analysis requires saved token IDs')
        comparisons = {m:sum(a['output_token_ids'] != b['output_token_ids']
                             for a,b in zip(records[method],records[m])) for m in ('baseline','swift')}
        row = {k:summaries[method][k] for k in ('metric_protocol','samples','tokens','inference_seconds',
               'tokens_per_second','global_tokens_per_second','mean_accepted_length','acceptance_rate')}
        row.update(tree_protocol=PROTOCOL,stage=args.stage,split=args.split,
                   layer_search=getattr(args,'layer_search','native'),model=model,seed=seed,dataset=task,method=method,
                   speedup_vs_baseline=base['speedup_ratio'],speedup_vs_swift=swift['speedup_ratio'],
                   total_time_speedup_vs_swift=swift['inference_time_speedup'],
                   token_mismatch_vs_baseline=comparisons['baseline'],
                   token_mismatch_vs_swift=comparisons['swift'])
        times = [sum(r['choices'][0]['wall_time']) for r in records[method]]
        row.update(p50_request_seconds=percentile(times,50),p95_request_seconds=percentile(times,95))
        metrics = [r.get('forward_metrics',{}) for r in records[method]]
        for name in ('controller_seconds','controlled_rounds','fallback_rounds','tree_buffer_hits','optimization_host_seconds'):
            row[name] = sum(m.get(name,0) for m in metrics)
        events = [e for m in metrics for e in m.get('tree_events',[])]
        labels = [l for e in events for l in e['rank_labels']]
        row.update(mean_tree_nodes=sum(e['verified_nodes'] for e in events)/len(events) if events else '',
                   mean_draft_depth=sum(e['draft_depth'] for e in events)/len(events) if events else '',
                   rank_labels=len(labels),rank_brier=sum(l['brier'] for l in labels)/len(labels) if labels else '',
                   learned_rank_labels=sum(l['support']>0 for l in labels),
                   raw_rank_brier=sum(l['raw_brier'] for l in labels)/len(labels) if labels else '',
                   rank_log_loss=sum(l['log_loss'] for l in labels)/len(labels) if labels else '')
        for phase in ('draft','draft_kv','candidate','verify','posterior','commit'):
            measured = any('phase_gpu_ms' in m for m in metrics)
            row[phase+'_gpu_seconds'] = sum(m.get('phase_gpu_ms',{}).get(phase,0) for m in metrics)/1000 if measured else ''
            row[phase+'_host_seconds'] = sum(m.get('phase_host_seconds',{}).get(phase,0) for m in metrics) if measured else ''
        if events:
            diagnostics=[]
            for record in records[method]:
                for index,event in enumerate(record.get('forward_metrics',{}).get('tree_events',[])):
                    diagnostics.append({'task_id':record['task_id'],'round':index,
                        **{k:event[k] for k in ('draft_depth','verified_nodes','committed','predicted_M','reason','cost_ready')},
                        'widths':json.dumps(event['widths']),'rank_labels':len(event['rank_labels'])})
            write_csv(folder/f'{task}.{method}.diagnostics.csv',diagnostics)
        if scores:
            score = scores[method]
            pair = compare_scores(score,scores['baseline'])
            write_json(folder/f'{task}.{method}.accuracy.json',dict(score,vs_baseline=pair))
            row.update(accuracy=score['accuracy'],correct=score['correct'],missing_answer_rate=score['missing_answer_rate'],
                       token_limit_rate=score['token_limit_rate'],correct_to_wrong_vs_baseline=len(pair['correct_to_wrong_ids']),
                       wrong_to_correct_vs_baseline=len(pair['wrong_to_correct_ids']))
        if actual_task == 'humaneval':
            score_path=folder/f'{task}.{method}.pass1.json'
            run([sys.executable,SCRIPTS/'evaluate_humaneval.py','--input',folder/f'{task}.{method}.jsonl',
                 '--samples',folder/f'{task}.{method}.samples.jsonl','--score',score_path],REPO)
            row['pass_at_1']=json.loads(score_path.read_text(encoding='utf-8'))['pass@1']
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temporary=path.with_suffix(path.suffix+'.tmp')
    with open(temporary,'w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage',choices=['diagnose','evaluate'],default='diagnose')
    parser.add_argument('--models',nargs='+',default=['meta-llama/Llama-2-7b-hf','meta-llama/Llama-2-13b-hf'])
    parser.add_argument('--code-models',nargs='+',default=['codellama/CodeLlama-7b-hf','codellama/CodeLlama-13b-hf'])
    parser.add_argument('--tasks',nargs='+',choices=['cnndm','gsm8k','humaneval'],default=['cnndm','gsm8k'])
    parser.add_argument('--gsm8k-profiles',nargs='+',choices=['paper64','quality512'],default=['paper64','quality512'])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[2024])
    parser.add_argument('--data-seed',type=int,default=2024)
    parser.add_argument('--data-num',type=int)
    parser.add_argument('--tree-max-nodes',type=int,default=251)
    parser.add_argument('--tree-lookahead',type=int,choices=[1,2],default=2)
    parser.add_argument('--tree-min-labels',type=int,default=32)
    parser.add_argument('--layer-search',choices=['native','fixed'],default='native',
                        help='fixed is a development-only initial-mask control, not paper SWIFT')
    parser.add_argument('--fixed-depths',nargs='+',type=int,default=[])
    parser.add_argument('--fixed-widths',nargs='+',type=int,default=[1,3,5])
    parser.add_argument('--swift-ref',default='main')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    args.split='train' if args.stage=='diagnose' else 'test'
    args.data_num=args.data_num if args.data_num is not None else (128 if args.stage=='diagnose' else 1000)
    if not {'baseline','swift'}.issubset(args.methods):
        parser.error('baseline and swift are required')
    if args.stage=='diagnose' and 'humaneval' in args.tasks:
        parser.error('HumanEval is evaluation-only')
    if args.stage=='evaluate' and args.layer_search!='native':
        parser.error('Evaluation keeps native layer search; fixed is diagnostic-only')
    if 'humaneval' in args.tasks and len(args.models)!=len(args.code_models):
        parser.error('Pair --models and --code-models by size')
    if not 1<=args.tree_max_nodes<=251 or args.data_num<1 or args.tree_min_labels<1:
        parser.error('Invalid budget/count/minimum')
    for sequence in (args.models,args.code_models,args.tasks,args.methods,args.seeds,args.gsm8k_profiles,args.fixed_depths,args.fixed_widths):
        if len(set(sequence))!=len(sequence):
            parser.error('Duplicate configuration values')
    if any(not 1<=d<=25 for d in args.fixed_depths) or any(not 1<=w<=10 for w in args.fixed_widths):
        parser.error('Fixed depths 1..25, widths 1..10 required')
    if any(1+d*w>args.tree_max_nodes for d in args.fixed_depths for w in args.fixed_widths):
        parser.error('Fixed sweep exceeds node budget')
    methods=args.methods+[f'fixed_d{d}_w{w}' for d in args.fixed_depths for w in args.fixed_widths]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=(args.output_dir or REPO/'_local/runs'/f'swift_tree_{stamp}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        native,controlled,source=prepare_sources(args,output)
        write_json(output/'run_config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   'methods':methods,'source':source,'protocol':PROTOCOL,
                   'note':'Greedy prototype; no new GPU results or SOTA claims. Native_control isolates timing/buffer instrumentation.'})
        shutil.copytree(SCRIPTS,output/'driver/scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(REPO/'model/swift_tree_control.py',output/'driver/swift_tree_control.py')
        if args.prepare_only:
            write_json(output/'status.json',{'status':'prepared_only','inference_ran':False})
            note(f'PREPARED: {output}; no model loading or GPU inference.')
            return
        check_environment(output)
        rows=[]
        for repetition,seed in enumerate(args.seeds):
            for task,actual_task,model,count,limit,top_p in configurations(args):
                model_id=model.rstrip('/').split('/')[-1]
                folder=output/f'seed_{seed}'/model_id
                folder.mkdir(parents=True,exist_ok=True)
                order=methods[repetition%len(methods):]+methods[:repetition%len(methods)]
                for method in order:
                    command=[sys.executable,'-m','evaluation_llama.'+('inference_baseline' if method=='baseline' else 'inference_swift'),
                             '--model-path',model,'--model-id',model_id,'--task-name',actual_task,'--data-num',count,
                             '--max-new-tokens',limit,'--temperature',0,'--top-p',top_p,'--dtype','float16',
                             '--seed',seed,'--data-seed',args.data_seed,'--research-split',args.split,
                             '--answer-file',folder/f'{task}.{method}.jsonl']
                    if method!='baseline':
                        command+=['--skip-ratio',.45]
                        if args.layer_search=='native':
                            command+=['--optimization','--bayes']
                    if method not in ('baseline','swift'):
                        policy='native' if method=='native_control' else method
                        if method.startswith('fixed_'):
                            _,d,w=method.split('_')
                            policy='fixed'
                            command+=['--fixed-depth',int(d[1:]),'--fixed-width',int(w[1:])]
                        command+=['--tree-policy',policy,'--tree-max-nodes',args.tree_max_nodes,
                                  '--tree-lookahead',args.tree_lookahead,'--tree-min-labels',args.tree_min_labels]
                    note(f'{args.stage}: {model_id}/{task}/{method}; seed {seed}; N={count}')
                    run(command,native if method in ('baseline','swift') else controlled)
                rows.extend(summarize(folder,task,actual_task,model,count,seed,args,methods))
                write_csv(output/'results.csv',rows)
        write_json(output/'status.json',{'status':'complete','stage':args.stage,'inference_ran':True,'rows':len(rows)})
        note(f'DONE: {output/"results.csv"}. Inspect controller overhead and mismatches before performance claims.')
    except Exception as error:
        write_json(output/'failure.json',{'status':'failed','error':repr(error)})
        raise


if __name__=='__main__':
    main()
