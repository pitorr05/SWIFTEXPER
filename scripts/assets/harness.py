"""Common research harness; the two algorithms live in separate checkouts."""
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from evaluation_llama.research_utils import METRIC_PROTOCOL, aggregate, digest, validate_output_path, write_json
from evaluation_llama.phase_timer import finish_request


def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize():
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, tree_length=250,
               max_output_length=4096, prompt_shots=None, device='cuda'):
    if task_name == 'cnndm':
        text = prompt_shots + 'Article: ' + prompt['article'] + '\nSummary:'
        tail_text = '\nSummary:'
    elif task_name == 'humaneval':
        text, tail_text = prompt['prompt'], ''
    else:
        raise ValueError(f'Unsupported task: {task_name}')
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    available = max_output_length - max_new_tokens - tree_length
    if available < 1:
        raise ValueError('Context window too small for generation and tree reserve')
    if ids.shape[1] > available:
        if task_name == 'humaneval':
            raise ValueError('Refusing to truncate a HumanEval prompt')
        tail = len(tokenizer(tail_text, add_special_tokens=False).input_ids)
        if available <= tail:
            raise ValueError('Insufficient room for CNN/DM prompt')
        ids = torch.cat((ids[:, :available-tail], ids[:, -tail:]), dim=1)
    return ids


def load_data(task_name, seed, data_num=10):
    if data_num <= 0:
        raise ValueError('data-num must be positive')
    if task_name == 'cnndm':
        from datasets import load_dataset
        dataset = load_dataset('cnn_dailymail', name='3.0.0', split='test')
        if data_num > len(dataset):
            raise ValueError('data-num exceeds CNN/DM test set')
        data = list(dataset.shuffle(seed=seed).select(range(data_num)))
        train = load_dataset('cnn_dailymail', name='3.0.0', split='train')
        shot = train.shuffle(seed=seed)[0]
        load_data.last_metadata = {'one_shot_id': shot['id'],
                                   'test_fingerprint': dataset._fingerprint,
                                   'train_fingerprint': train._fingerprint}
        shots = 'Article: ' + shot['article'] + '\nSummary: ' + shot['highlights'].replace('\n', '') + '\n'
        return data, shots
    if task_name == 'humaneval':
        from human_eval.data import read_problems
        data = list(read_problems().values())
        if data_num > len(data):
            raise ValueError(f'HumanEval has {len(data)} tasks, requested {data_num}')
        load_data.last_metadata = {'problems_sha256': digest(data)}
        return data[:data_num], ''
    raise ValueError(f'Unsupported task: {task_name}')


def source_metadata(config, model):
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    files = sorted(p for folder in ('evaluation_llama', 'model/swift') for p in (root / folder).glob('*.py'))
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'datasets', 'bayesian-optimization', 'human-eval'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'schema_version': 1, 'metric_protocol': METRIC_PROTOCOL, 'config': config, 'source': provenance,
            'source_sha256': {str(p.relative_to(root)): digest(p.read_text(encoding='utf-8')) for p in files},
            'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
            'cuda_runtime': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'model_revision': getattr(model.config, '_commit_hash', None), 'model_config': model.config.to_dict(),
            'timing': 'synchronized forward wall time: includes prefill/cache/optimization, excludes tokenization and output I/O',
            'warmup': 'disabled; cold stream, cache empty at start'}


@torch.inference_mode()
def run_eval(model, tokenizer, forward_func, model_id, answer_file, max_new_tokens,
             num_gpus_per_model, num_gpus_total, task_name, data_num, seed,
             data_seed=2024, warmup_runs=0, run_config=None, **kwargs):
    if num_gpus_total != 1 or num_gpus_per_model != 1:
        raise ValueError('Protocol requires one GPU and batch size 1')
    if max_new_tokens <= 0 or warmup_runs != 0:
        raise ValueError('Positive token budget required; this protocol does not use warmup')
    answer_file = validate_output_path(answer_file)
    Path(answer_file).parent.mkdir(parents=True, exist_ok=True)
    with open(answer_file, 'x', encoding='utf-8') as stream:
        model.eval()
        seed_everything(seed)
        data, shots = load_data(task_name, data_seed, data_num)
        manifest = {'task_name': task_name, 'data_seed': data_seed,
                    'dataset': 'cnn_dailymail/3.0.0/test' if task_name == 'cnndm' else 'human_eval/read_problems',
                    'one_shot_prompt': shots, 'one_shot_sha256': digest(shots),
                    'task_ids': [q['task_id'] if task_name == 'humaneval' else q['id'] for q in data],
                    'dataset_metadata': getattr(load_data, 'last_metadata', {})}
        write_json(answer_file + '.manifest.json', manifest)
        metadata = source_metadata(run_config or {}, model)
        metadata['manifest_sha256'] = digest(manifest)
        write_json(answer_file + '.metadata.json', metadata)
        device = next(model.parameters()).device
        records = []
        for question in tqdm(data):
            task_id = question['task_id'] if task_name == 'humaneval' else question['id']
            forward_func.request_context = {'task_id': task_id}
            ids = clip_input(tokenizer, question, task_name, max_new_tokens=max_new_tokens,
                             prompt_shots=shots, max_output_length=model.config.max_position_embeddings, device=device)
            synchronize()
            start = time.perf_counter()
            output_ids, reported_tokens, step, lengths, draft_num = forward_func(
                ids, model, tokenizer, max_new_tokens, **kwargs)
            synchronize()
            elapsed = time.perf_counter() - start
            metrics = dict(getattr(forward_func, 'last_metrics', {}))
            if 'statistics' in kwargs:
                metrics.update(finish_request(kwargs['statistics']))
                metrics['opt_iter_total'] = kwargs['statistics']['opt_iter']
                metrics['origin_score'] = kwargs['statistics']['origin_score']
                metrics['optimization_active'] = kwargs['statistics']['optimization']
                metrics['incumbent_layers'] = [list(map(int, x)) for x in model.get_skip_layers()]
            tokens = output_ids[0, ids.shape[1]:].tolist()
            if len(tokens) != reported_tokens or len(tokens) > max_new_tokens:
                raise RuntimeError('Invalid forward token accounting or budget')
            eos = tokenizer.eos_token_id
            if eos in tokens and tokens.index(eos) != len(tokens)-1:
                raise RuntimeError('Forward returned tokens after EOS')
            text = tokenizer.decode(tokens, skip_special_tokens=True, spaces_between_special_tokens=False)
            accepted = sum(max(0, int(n)-1) for n in lengths)
            choice = {'turns': text, 'decoding_steps': [int(step)], 'new_tokens': [len(tokens)],
                      'wall_time': [elapsed], 'accept_lengths': list(map(int, lengths)),
                      'draft_token_num': int(draft_num), 'accepted_draft_tokens': accepted,
                      'acceptance_rate': accepted/draft_num if draft_num else None}
            inputs = ids[0].tolist()
            row = {'schema_version': 1, 'task_name': task_name, 'task_id': task_id, 'model_id': model_id,
                   'input_token_ids': inputs, 'input_sha256': digest(inputs), 'output_token_ids': tokens,
                   'choices': [choice], 'forward_metrics': metrics, 'tstamp': time.time()}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            records.append(row)
        summary = aggregate(records)
        summary['status'] = 'complete'
        write_json(answer_file + '.summary.json', summary)
        print(json.dumps(summary, indent=2))
