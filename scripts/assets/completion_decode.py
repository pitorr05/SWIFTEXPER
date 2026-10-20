"""Decode a saved continuation in its original prompt context, without code repair."""

POLICY = 'prompt_context_decode_v1'


def decode_continuation(tokenizer, input_ids, output_ids, expected_prompt=None):
    options = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False,
                   spaces_between_special_tokens=False)
    prefix = tokenizer.decode(input_ids, **options)
    if expected_prompt is not None and prefix != expected_prompt:
        raise ValueError('Decoded input differs from the original HumanEval prompt; '
                         'check tokenizer revision and prompt provenance before scoring')
    full = tokenizer.decode(list(input_ids) + list(output_ids), **options)
    if not full.startswith(prefix):
        raise ValueError('Tokenizer changed the decoded prompt at the continuation boundary; '
                         'refusing to guess or repair indentation')
    return full[len(prefix):]
