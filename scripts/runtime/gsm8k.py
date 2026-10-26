"""GSM8K protocol helpers; scoring needs only the Python standard library."""
from decimal import Decimal, InvalidOperation
import re

PROTOCOL = 'gsm8k_5shot_strict_hash_v1'
FINAL = re.compile(r'####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\w,/]|\.\d)')


def extract_answer(text):
    # Use the first completed solution, ignoring any subsequently generated Q&A.
    text = text.split('\nQuestion:', 1)[0]
    match = FINAL.search(text)
    if match is None:
        return None
    try:
        number = Decimal(match.group(1).replace(',', ''))
        return '0' if number == 0 else str(number.normalize())
    except InvalidOperation:
        return None


def load_gsm8k(seed, count):
    from datasets import load_dataset
    test = load_dataset('openai/gsm8k', 'main', split='test')
    train = load_dataset('openai/gsm8k', 'main', split='train')
    if not 0 < count <= len(test):
        raise ValueError('data-num exceeds GSM8K test set or is not positive')
    # Assign IDs before shuffling: IDs refer to original split row positions.
    test = test.add_column('id', [f'gsm8k/test/{i}' for i in range(len(test))])
    train = train.add_column('id', [f'gsm8k/train/{i}' for i in range(len(train))])
    shots = list(train.shuffle(seed=seed).select(range(5)))
    data = list(test.shuffle(seed=seed).select(range(count)))
    prefix = 'Solve each problem. End your solution with #### followed by the final numeric answer.\n\n'
    for shot in shots:
        prefix += f'Question: {shot["question"]}\nAnswer: {shot["answer"]}\n\n'
    metadata = {'protocol': PROTOCOL, 'dataset': 'openai/gsm8k/main/test',
                'test_fingerprint': test._fingerprint, 'train_fingerprint': train._fingerprint,
                'few_shot_ids': [s['id'] for s in shots]}
    return data, prefix, metadata


def score_records(records, expected_count):
    if len(records) != expected_count or len({r['task_id'] for r in records}) != expected_count:
        raise ValueError('Incomplete or duplicated GSM8K records')
    details = []
    for row in records:
        if row.get('task_name') != 'gsm8k' or row.get('gsm8k_protocol') != PROTOCOL:
            raise ValueError('Unexpected GSM8K task or scoring protocol')
        gold = extract_answer(row['reference_answer'])
        if gold is None:
            raise ValueError('Invalid GSM8K reference answer')
        prediction = extract_answer(row['choices'][0]['turns'])
        if type(row.get('hit_token_limit')) is not bool:
            raise ValueError('Missing token-limit accounting')
        details.append({'task_id': row['task_id'], 'gold': gold, 'prediction': prediction,
                        'correct': prediction == gold, 'hit_token_limit': row['hit_token_limit']})
    correct = sum(d['correct'] for d in details)
    missing = sum(d['prediction'] is None for d in details)
    capped = sum(d['hit_token_limit'] for d in details)
    return {'status': 'complete', 'protocol': PROTOCOL, 'total': expected_count,
            'correct': correct, 'accuracy': correct / expected_count,
            'missing_answer_count': missing, 'missing_answer_rate': missing / expected_count,
            'token_limit_count': capped, 'token_limit_rate': capped / expected_count,
            'details': details}


def compare_scores(score, baseline):
    left, right = score['details'], baseline['details']
    if [(d['task_id'], d['gold']) for d in left] != [(d['task_id'], d['gold']) for d in right]:
        raise ValueError('GSM8K score IDs or references differ')
    return {'correct_to_wrong_ids': [b['task_id'] for a, b in zip(left, right) if b['correct'] and not a['correct']],
            'wrong_to_correct_ids': [b['task_id'] for a, b in zip(left, right) if a['correct'] and not b['correct']]}
