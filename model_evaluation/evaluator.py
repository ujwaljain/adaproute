"""Compare fixed Gemini fallback policies using saved LiveCodeBench responses."""

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import re

from accounting import FLASH, LITE, MODELS, PRICES, STRONG, estimate
import assets
import benchmark
from common import digest, load, save

HERE = Path(__file__).resolve().parent
POLICIES = {
    'always_flash_lite': (LITE,),
    'always_3_flash': (FLASH,),
    'always_3_8_flash': (STRONG,),
    'lite_then_3_8': (LITE, STRONG),
    'flash_then_3_8': (FLASH, STRONG),
    'lite_then_flash_then_3_8': (LITE, FLASH, STRONG),
}
BASELINE = 'always_3_8_flash'


def validate_response(record, job):
    """Check that a received answer belongs to the expected question and Gemini model."""
    if any(record.get(key) != job[key] for key in ('job_id', 'question_key', 'model', 'request')):
        raise ValueError('Saved response does not match its question/model request')
    response = record['response']
    choices = response.get('choices', [])
    if response.get('model') != job['model'] or 'error' in response or len(choices) != 1:
        raise ValueError('Invalid received Gemini response')
    choice = choices[0]
    content = choice.get('message', {}).get('content')
    if ('error' in choice or choice.get('finish_reason') not in ('stop', 'length')
            or not isinstance(content, str) or not content.strip()):
        raise ValueError('Invalid response content or finish reason')


def load_inputs(source, questions):
    """Load matching questions and responses, excluding whole incomplete questions.

    Every policy uses the same questions. Received truncated answers stay in
    the comparison; only missing or unsuccessful API responses cause exclusion.
    Private test strings are loaded as data but are not decoded here.
    """
    raw = (source/'manifest.json').read_bytes()
    if digest(raw) != (source/'manifest.sha256').read_text().strip():
        raise ValueError('Source manifest checksum mismatch')
    manifest = json.loads(raw)
    raw = questions.read_bytes()
    if digest(raw) != manifest['questions_sha256']:
        raise ValueError('Questions differ from the collected dataset')
    rows = json.loads(raw)
    samples = {row['platform']+':'+row['question_id']: row for row in rows}
    if not samples or len(samples) != len(rows) or len(rows) != manifest['question_count']:
        raise ValueError('Empty, duplicate or mismatched questions')
    if {model['id'] for model in manifest['config']['models']} != set(MODELS):
        raise ValueError('Expected the three Gemini models from this experiment')
    records, excluded, seen, ids, messages = {}, {}, set(), set(), {}
    for job in manifest['jobs']:
        key, model, job_id = job['question_key'], job['model'], job['job_id']
        if (key not in samples or model not in MODELS or (key, model) in seen
                or job_id in ids or not re.fullmatch(r'q\d{3}-m\d{2}', job_id)):
            raise ValueError('Duplicate or invalid source job')
        seen.add((key, model))
        ids.add(job_id)
        if job['request']['model'] != model:
            raise ValueError('Request model differs from its job')
        prompt = job['request']['messages']
        if key in messages and messages[key] != prompt:
            raise ValueError('Models received different question prompts')
        messages[key] = prompt
        path = source/'records'/(job_id+'.json')
        raw = path.read_bytes() if path.exists() else None
        record = json.loads(raw) if raw is not None else None
        if record is None or record.get('status') != 'received':
            excluded.setdefault(key, []).append({'model': model, 'status': record.get('status') if record else 'missing'})
            continue
        validate_response(record, job)
        record['record_sha256'] = digest(raw)
        record['cost'] = estimate(record)
        records.setdefault(key, {})[model] = record
    if seen != {(key, model) for key in samples for model in MODELS}:
        raise ValueError('Source manifest is missing question/model jobs')
    included = [key for key in samples if key not in excluded]
    if not included:
        raise ValueError('No questions have responses from all three models')
    coverage = {'source_questions': len(samples), 'evaluated_questions': len(included),
                'source_received_responses': sum(len(pool) for pool in records.values()),
                'excluded_questions': excluded,
                'source_truncated_jobs': [r['job_id'] for pool in records.values() for r in pool.values()
                                          if r['response']['choices'][0]['finish_reason'] == 'length']}
    return {key: samples[key] for key in included}, {key: records[key] for key in included}, coverage


def cached_grade(record, sample, phase, output, checker_key):
    """Reuse a verdict only for the same response, tests and checker.

    Changes to prices or fallback policies do not require grading again.
    An infrastructure failure is saved for inspection and retried on the next
    run; it never becomes a failed solution or a fallback trigger.
    """
    path = output/phase/(record['job_id']+'.json')
    cache_key = digest((record['record_sha256'] + sample['input_output'] + checker_key).encode())
    if path.exists():
        result = load(path)
        if (result.get('cache_key') == cache_key and type(result.get('success')) is int
                and result['success'] in (0, 1)):
            return result
    result = benchmark.evaluate(record['response'], sample)
    result['cache_key'] = cache_key
    save(path, result)
    if type(result['success']) is not int or result['success'] not in (0, 1):
        raise RuntimeError(f"Checker unavailable for {record['job_id']}: {result['status']}; fix it and rerun")
    print(f"{phase} {record['job_id']}: {result['status']}, passed={result['success']}", flush=True)
    return result


def grade_responses(samples, records, phase, output, checker_key):
    """Grade public examples or full tests, decoding hidden tests only in the full phase."""
    (output/phase).mkdir(exist_ok=True)
    verdicts = {}
    for key, row in samples.items():
        sample = benchmark.sample(row['public_test_cases'], json.loads(row['metadata']).get('func_name'),
                                  row['private_test_cases'] if phase == 'full' else None)
        for record in records[key].values():
            verdicts[record['job_id']] = cached_grade(record, sample, phase, output, checker_key)
    return verdicts


def choose_attempts(pool, public, order):
    """Keep the first public-passing answer, or the final model if all checks fail."""
    attempts = []
    for model in order:
        job_id = pool[model]['job_id']
        attempts.append(job_id)
        if len(order) == 1 or public[job_id]['success'] == 1:
            break
    return attempts


def decide(records, public):
    """Choose all fixed policy paths using public verdicts, without hidden grades."""
    return [{'question_key': key, 'policy': name, 'attempted_job_ids': choose_attempts(pool, public, order)}
            for key, pool in records.items() for name, order in POLICIES.items()]


def summarize(records, public, full, decisions, coverage):
    """Sum every attempted model's cost and compare final checker scores with 3.8 Flash."""
    jobs = {r['job_id']: r for pool in records.values() for r in pool.values()}
    entries = []
    for decision in decisions:
        ids = decision['attempted_job_ids']
        final = ids[-1]
        entries.append({'question_key': decision['question_key'], 'policy': decision['policy'],
                        'attempted_models': [jobs[j]['model'] for j in ids],
                        'final_model': jobs[final]['model'], 'passed': full[final]['success'],
                        'fallback': int(len(ids) > 1),
                        'estimated_cost_usd': str(sum((Decimal(jobs[j]['cost']['estimated_cost_usd']) for j in ids), Decimal(0))),
                        'input_tokens': sum(jobs[j]['cost']['input_tokens'] for j in ids),
                        'output_tokens_including_thinking': sum(jobs[j]['cost']['output_tokens_including_thinking'] for j in ids),
                        'false_public_accept': int(len(POLICIES[decision['policy']]) > 1
                                                   and public[final]['success'] == 1 and full[final]['success'] == 0)})
    baseline = sum((Decimal(r['estimated_cost_usd']) for r in entries if r['policy'] == BASELINE), Decimal(0))
    policies = {}
    for name in POLICIES:
        rows = [r for r in entries if r['policy'] == name]
        cost = sum((Decimal(r['estimated_cost_usd']) for r in rows), Decimal(0))
        policies[name] = {'questions': len(rows), 'passed': sum(r['passed'] for r in rows),
                          'estimated_cost_usd': str(cost), 'savings_percent': float(100 * (1 - cost/baseline)),
                          'fallbacks': sum(r['fallback'] for r in rows),
                          'false_public_accepts': sum(r['false_public_accept'] for r in rows)}
    return {'coverage': coverage, 'prices_usd_per_million_input_output': PRICES,
            'policies': policies, 'per_question': entries,
            'note': 'Offline generation cost estimates including thinking and rejected answers; not actual billing savings. '
                    'Scores use the pinned checker, which can reject valid floating-point or alternative outputs. '
                    'Collection HTTP retries and local checker compute are not priced.'}


def run(source, questions, output):
    """Verify public examples, save fallback decisions, then score hidden tests and costs."""
    source, questions, output = Path(source).resolve(), Path(questions).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from the saved model run')
    samples, records, coverage = load_inputs(source, questions)
    checker = assets.verify()
    checker_key = digest(json.dumps(checker, sort_keys=True).encode()
                         + (HERE/'benchmark.py').read_bytes() + (HERE/'sandbox_worker.py').read_bytes())
    benchmark.selftest()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Comparing {coverage['evaluated_questions']}/{coverage['source_questions']} questions.", flush=True)
    for key in coverage['excluded_questions']:
        print('Excluded incomplete question:', key, flush=True)
    public = grade_responses(samples, records, 'public', output, checker_key)
    decisions = decide(records, public)
    save(output/'decisions.json', decisions)
    full = grade_responses(samples, records, 'full', output, checker_key)
    report = summarize(records, public, full, decisions, coverage)
    save(output/'summary.json', report)
    with (output/'policy-results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['per_question'][0]))
        writer.writeheader()
        for row in report['per_question']:
            writer.writerow({**row, 'attempted_models': ' -> '.join(row['attempted_models'])})
    for name, result in report['policies'].items():
        print(f"{name}: {result['passed']}/{result['questions']} passed, "
              f"${result['estimated_cost_usd']}, savings={result['savings_percent']:.2f}%")
    return report


def main():
    """Run the Gemini experiment; optional arguments only select input/output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=HERE.parent/'model_runner/runs/gemini-30-questions')
    parser.add_argument('--questions', type=Path, default=HERE.parent/'data_fetcher/data/questions.json')
    parser.add_argument('--output-dir', type=Path, default=HERE/'runs/gemini-fallback-simple')
    args = parser.parse_args()
    try:
        run(args.source_run, args.questions, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
