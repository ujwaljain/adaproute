"""Undiscounted USD costs for this experiment's three Gemini models."""

from decimal import Decimal

LITE = 'gemini-3.1-flash-lite'
FLASH = 'gemini-3-flash-preview'
STRONG = 'gemini-3.8-flash'
MODELS = (LITE, FLASH, STRONG)

# USD per million tokens: input, then output including thinking.
PRICES = {LITE: ('0.25', '1.50'), FLASH: ('0.50', '3.00'), STRONG: ('1.50', '7.50')}


def estimate(record):
    """Price a saved Gemini response, including its thinking tokens.

    Gemini's total_tokens includes prompt, answer and thinking tokens. Subtract
    prompt_tokens to obtain all generated tokens; completion_tokens alone can
    omit thinking. Missing or inconsistent usage is an error, not zero cost.
    """
    usage = record['response'].get('usage', {})
    counts = [usage.get(name) for name in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError('Missing or invalid Gemini token usage')
    prompt, completion, total = counts
    if prompt == 0 or total < prompt + completion:
        raise ValueError('Inconsistent Gemini token usage')
    generated = total - prompt
    input_rate, output_rate = map(Decimal, PRICES[record['model']])
    cost = (prompt * input_rate + generated * output_rate) / Decimal(1_000_000)
    return {'input_tokens': prompt, 'output_tokens_including_thinking': generated,
            'estimated_cost_usd': str(cost)}
