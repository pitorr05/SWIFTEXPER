"""Rebuild research patches from immutable local Git snapshots. No network."""
from pathlib import Path
import shutil
import subprocess

ROOT = REFERENCE = ADAPTIVE = None
ORIGINALS = {}


def original(root, path):
    return ORIGINALS[(str(root), path)]


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one match for {old[:100]!r}, got {text.count(old)}')
    return text.replace(old, new, 1)


def write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Python 3.9 is the original SWIFT environment; Path.write_text(newline=...) is newer.
    with open(target, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


def patch_cli(text, swift):
    text = replace(text, 'from evaluation_llama.eval import run_eval',
                   'from evaluation_llama.eval import run_eval\nfrom evaluation_llama.research_utils import validate_output_path')
    text = replace(text, '    args = parser.parse_args()', '''    parser.add_argument('--data-seed', type=int, default=2024)
    parser.add_argument('--warmup-runs', type=int, default=0)
    parser.add_argument('--model-revision', default=None)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.data_num < 1 or args.warmup_runs < 0:
        parser.error('Positive token budget/data count and nonnegative warmup count required')
    if args.task_name not in ('cnndm', 'humaneval', 'gsm8k'):
        parser.error('Supported tasks: cnndm, humaneval, gsm8k')
    if args.task_name == 'humaneval' and args.data_num > 164:
        parser.error('HumanEval contains 164 tasks')
    if args.num_gpus_total != 1 or args.num_gpus_per_model != 1:
        parser.error('Research protocol supports one visible GPU')''')
    prefix = 'outputs' if swift else 'test'
    old = f'    answer_file = f"{prefix}/{{args.task_name}}/{{args.task_name}}_{{args.data_num}}/model_answer/{{args.model_id}}/{{args.model_name}}.jsonl"'
    text = replace(text, old, old.replace('answer_file = f', 'answer_file = args.answer_file or f') + '\n    answer_file = validate_output_path(answer_file)')
    text = replace(text, '        device_map="auto"', '        revision=args.model_revision,\n        device_map="auto"')
    text = replace(text, '    tokenizer = AutoTokenizer.from_pretrained(args.model_path)',
                   '    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.model_revision)')
    text = replace(text, '        seed=args.seed,', '        seed=args.seed,\n        data_seed=args.data_seed,\n        warmup_runs=args.warmup_runs,\n        run_config=vars(args),')
    return text


def patch_swift(text, function_name='swift_forward'):
    text = replace(text, 'import argparse', 'import argparse\nfrom evaluation_llama.phase_timer import begin_request, optimization_timer')
    text = replace(text, '    # Avoid modifying the input_ids in-place', '''    if max_new_tokens < 1 or max_steps < max_new_tokens:
        raise ValueError('Positive budget and max_steps >= max_new_tokens required')
    begin_request(statistics)
    verified_tree_tokens = 0
    verified_accepted_draft_tokens = 0
    # Avoid modifying the input_ids in-place''')
    text = replace(text, '        logits, outputs = tree_decoding(',
                   '        verified_tree_tokens += int(tree_candidates.numel())\n        logits, outputs = tree_decoding(')
    text = replace(text, '        input_ids, new_token_num, sample_token = update_inference_inputs(', '''        verified_accepted_draft_tokens += int(accept_length)
        # Commit only the verified prefix within the budget, stopping at first EOS.
        committed = min(int(accept_length) + 1, max_new_tokens - new_token_num)
        prefix = candidates[best_candidate, :committed].tolist()
        if tokenizer.eos_token_id in prefix:
            committed = prefix.index(tokenizer.eos_token_id) + 1
        accept_length = committed - 1
        input_ids, new_token_num, sample_token = update_inference_inputs(''')
    old = '''        # layer set optimization
'''
    text = replace(text, old, '''        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length += accept_length_tree
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist() or new_token_num >= max_new_tokens:
            break

        # layer set optimization
''')
    start = text.index('            swift_optimization(')
    end = text.index('                utility=utility)', start) + len('                utility=utility)')
    block = text[start:end]
    text = text[:start] + '            with optimization_timer(statistics, model):\n' + '\n'.join('    '+line for line in block.splitlines()) + text[end:]
    start = text.index('        accept_length_tree = input_ids.shape[1] - cur_length', text.index('        # swift drafting'))
    end = text.index('    logging.info("token acceptance rate:', start)
    text = text[:start] + text[end:]
    text = replace(text, '    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num', '''    swift_forward.last_metrics = {
        'verified_tree_tokens': verified_tree_tokens,
        'verified_accepted_draft_tokens': verified_accepted_draft_tokens,
        'draft_steps': draft_token_num,
    }
    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num''')
    if function_name != 'swift_forward':
        text = replace(text, '    swift_forward.last_metrics = {',
                       f'    {function_name}.last_metrics = {{')
    return patch_cli(text, True)


def main():
    # Helpers are copied byte-for-byte; no import path can route reference through contribution code.
    for root in (REFERENCE, ADAPTIVE):
        for source, name in (('harness.py', 'eval.py'), ('speed.py', 'speed.py'), ('research_utils.py', 'research_utils.py'), ('phase_timer.py', 'phase_timer.py'), ('gsm8k.py', 'gsm8k.py'), ('completion_decode.py', 'completion_decode.py')):
            shutil.copyfile(ROOT / source, root / 'evaluation_llama' / name)
    baseline = patch_cli(original(REFERENCE, 'evaluation_llama/inference_baseline.py'), False)
    # Match the eager target attention backend used by the custom SWIFT model.
    baseline = replace(baseline, '        revision=args.model_revision,',
                       '        attn_implementation="eager",\n        revision=args.model_revision,')
    swift = patch_swift(original(REFERENCE, 'evaluation_llama/inference_swift.py'))
    for root in (REFERENCE, ADAPTIVE):
        write(root, 'evaluation_llama/inference_baseline.py', baseline)
    write(REFERENCE, 'evaluation_llama/inference_swift.py', swift)
    adaptive = patch_swift(original(ADAPTIVE, 'evaluation_llama/inference_swift.py'),
                           function_name='_swift_forward_impl')
    write(ADAPTIVE, 'evaluation_llama/inference_swift.py', adaptive)



def prepare(reference, adaptive, assets):
    global ROOT, REFERENCE, ADAPTIVE, ORIGINALS
    ROOT, REFERENCE, ADAPTIVE = assets, reference, adaptive
    ORIGINALS = {}
    for folder in (reference, adaptive):
        for path in ('evaluation_llama/inference_baseline.py', 'evaluation_llama/inference_swift.py', 'model/swift/utils.py'):
            ORIGINALS[(str(folder), path)] = (folder / path).read_text(encoding='utf-8')
    if not (adaptive / 'evaluation_llama/adaptive_cache.py').is_file() or 'def _swift_forward_impl(' not in original(adaptive, 'evaluation_llama/inference_swift.py'):
        raise ValueError('Adaptive ref is the old implementation. Push the fixed AdaptiveCache branch and fetch it on the server, or select --adaptive-ref AdaptiveCache locally.')
    for path in ('model/swift/utils.py', 'model/swift/modeling_llama.py', 'model/swift/kv_cache.py'):
        if (reference/path).read_bytes() != (adaptive/path).read_bytes():
            raise ValueError('Model implementations differ between refs; review before applying the common decoder')
    main()
