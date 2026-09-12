"""Tests for the fixed Gemini cost and fallback experiment."""

import base64
from contextlib import ExitStack, redirect_stdout
from decimal import Decimal
import io
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

import accounting
from accounting import LITE, FLASH, STRONG, MODELS
import benchmark
import evaluator
from common import digest, load, save


def fixture(root):
    """Create a saved two-question Gemini run without model API calls."""
    source, output, questions = root/'source', root/'evaluation', root/'questions.json'
    (source/'records').mkdir(parents=True)
    rows = [{'platform': 'fixture', 'question_id': str(q),
             'public_test_cases': json.dumps([{'input': f'public:{q}', 'output': '2'}]),
             'private_test_cases': json.dumps([{'input': f'PRIVATE_SECRET:{q}', 'output': '2'}]),
             'metadata': '{}'} for q in range(2)]
    save(questions, rows)
    jobs = []
    for q in range(2):
        for m, model in enumerate(MODELS):
            job_id = f'q{q:03d}-m{m:02d}'
            job = {'job_id': job_id, 'question_key': f'fixture:{q}', 'model': model,
                   'request': {'model': model, 'messages': [{'role': 'user', 'content': f'question {q}'}]}}
            jobs.append(job)
            save(source/'records'/(job_id+'.json'), {**job, 'status': 'received',
                 'response': {'model': model, 'choices': [{'finish_reason': 'stop',
                    'message': {'content': f'```python\n# {job_id}\nprint(2)\n```'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 150}}})
    save(source/'manifest.json', {'question_count': 2, 'jobs': jobs,
         'questions_sha256': digest(questions.read_bytes()),
         'config': {'models': [{'id': m} for m in MODELS]}})
    (source/'manifest.sha256').write_text(digest((source/'manifest.json').read_bytes())+'\n')
    return source, questions, output


def fake_evaluate(response, sample):
    """Simulate false public acceptance and a question requiring the strong model."""
    full = 'PRIVATE_SECRET' in sample['input_output']
    question_zero = 'q000-' in response['choices'][0]['message']['content']
    model = response['model']
    success = int(model == STRONG or (question_zero and (model == FLASH or not full)))
    return {'status': 'scored', 'success': success, 'verification_seconds': 0.25}


class AccountingTests(unittest.TestCase):
    """Check the fixed Gemini rates and thinking-inclusive token calculation."""

    def test_cost_includes_thinking_once(self):
        """Price 50 generated tokens rather than only the 20 visible tokens."""
        record = {'model': STRONG, 'response': {'usage': {
            'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 150}}}
        result = accounting.estimate(record)
        self.assertEqual(result['output_tokens_including_thinking'], 50)
        self.assertEqual(Decimal(result['estimated_cost_usd']), Decimal('0.000525'))

    def test_missing_or_inconsistent_usage_is_not_zero_cost(self):
        """Reject absent, boolean, negative and inconsistent usage counts."""
        for usage in ({}, {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2},
                      {'prompt_tokens': 5, 'completion_tokens': 4, 'total_tokens': 7},
                      {'prompt_tokens': -1, 'completion_tokens': 2, 'total_tokens': 3}):
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                accounting.estimate({'model': LITE, 'response': {'usage': usage}})


class BenchmarkTests(unittest.TestCase):
    """Check decoding boundaries and fail-closed execution behavior."""

    def test_private_decoder_rejects_executable_pickle(self):
        """Decode the dataset format while refusing global resolution."""
        tests = [{'input': '1', 'output': '2'}]
        encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
        self.assertEqual(benchmark.decode_private(encoded), tests)
        malicious = base64.b64encode(zlib.compress(b"cos\nsystem\n(S'false'\ntR.")).decode()
        with self.assertRaisesRegex(ValueError, 'globals'):
            benchmark.decode_private(malicious)

    def test_public_samples_whitelist_only_public_fields(self):
        """Strip extra fields and never decode hidden tests during public setup."""
        with patch('benchmark.decode_private', side_effect=AssertionError('private access')):
            sample = benchmark.sample('[{"input":"1","output":"2","private":"SECRET"}]', 'solve')
        self.assertNotIn('SECRET', json.dumps(sample))
        self.assertEqual(json.loads(sample['input_output'])['fn_name'], 'solve')

    def test_truncation_and_ambiguous_code_reject_without_execution(self):
        """Incomplete outputs should trigger fallback and still retain generation cost."""
        with patch('benchmark.grade_sample') as grade:
            for finish, content in [('length', '```python\nprint(1)\n```'),
                                    ('stop', 'no code'), ('stop', '```python\na\n```\n```python\nb\n```')]:
                response = {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
                self.assertEqual(benchmark.evaluate(response, {})['success'], 0)
            grade.assert_not_called()

    def test_missing_bubblewrap_never_executes_on_host(self):
        """Require OS isolation rather than silently falling back to subprocess Python."""
        with patch('benchmark.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'refusing host'):
                benchmark.command()

    def test_negative_checker_codes_are_failures_and_crashes_unknown(self):
        """Negative error codes are truthy in Python but are not passing tests."""
        def execute(command, **kwargs):
            """Return a synthetic checker error without running generated code."""
            kwargs['stdout'].write(b'{"results":[-4],"metadata":{}}')
            return subprocess.CompletedProcess(command, 0)
        with patch('benchmark.command', return_value=['fake']), patch('benchmark.subprocess.run', side_effect=execute):
            self.assertEqual(benchmark.grade_sample({}, 'not executed')['success'], 0)
        with patch('benchmark.command', return_value=['fake']), patch('benchmark.subprocess.run', return_value=subprocess.CompletedProcess([], 1)):
            self.assertIsNone(benchmark.grade_sample({}, 'not executed')['success'])

    def test_sandbox_command_does_not_mount_repository_or_inherit_keys(self):
        """The child sees only its checker, dependencies and system runtime."""
        with patch('benchmark.shutil.which', return_value='/usr/bin/bwrap'):
            command = benchmark.command()
        self.assertIn('--unshare-all', command)
        self.assertIn('--clearenv', command)
        self.assertNotIn(str(evaluator.HERE.parent), command)
        self.assertNotIn('.env', command)

    def test_incomplete_or_garbage_checker_results_stay_unknown(self):
        """Do not report correctness when the grader returns malformed or partial data."""
        sample = benchmark.sample('[{"input":"1","output":"2"},{"input":"2","output":"3"}]')
        for results in ([True], ['garbage']):
            def execute(command, **kwargs):
                """Supply a malformed checker reply without executing code."""
                kwargs['stdout'].write(json.dumps({'results': results}).encode())
                return subprocess.CompletedProcess(command, 0)
            with patch('benchmark.command', return_value=['fake']), patch('benchmark.subprocess.run', side_effect=execute):
                self.assertIsNone(benchmark.grade_sample(sample, 'not executed')['success'])


class ReplayTests(unittest.TestCase):
    """Exercise the complete single-command workflow on synthetic responses."""

    def setUp(self):
        """Create private fixtures and replace sandbox execution with known verdicts."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source, self.questions, self.output = fixture(root)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch('assets.verify', return_value={'checker': 'fixture'}))
        self.stack.enter_context(patch('benchmark.selftest'))
        self.grader = self.stack.enter_context(patch('benchmark.evaluate', side_effect=fake_evaluate))
        self.network = self.stack.enter_context(patch('assets.urlopen', side_effect=AssertionError('Unexpected network call')))

    def run_evaluation(self):
        """Run the production evaluation workflow on this fixture."""
        return evaluator.run(self.source, self.questions, self.output)

    def test_costs_include_rejected_attempts_and_false_accepts(self):
        """Keep all attempt costs and grade final selections using hidden tests."""
        before = {p.name: p.read_bytes() for p in (self.source/'records').glob('*.json')}
        report = self.run_evaluation()
        lite = report['policies']['lite_then_3_8']
        self.assertEqual(lite['passed'], 1)
        self.assertEqual(lite['fallbacks'], 1)
        self.assertEqual(lite['false_public_accepts'], 1)
        self.assertEqual(Decimal(lite['estimated_cost_usd']), Decimal('0.000725'))
        self.assertEqual(report['policies']['always_3_8_flash']['passed'], 2)
        self.assertEqual(report['policies']['flash_then_3_8']['passed'], 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.source/'records').glob('*.json')})
        self.network.assert_not_called()

    def test_resume_and_price_changes_reuse_grades(self):
        """Report changes must not force repeated code execution."""
        first = self.run_evaluation()
        calls = self.grader.call_count
        self.assertEqual(self.run_evaluation(), first)
        with patch.dict(accounting.PRICES, {LITE: ('0.25', '3.00')}):
            changed = self.run_evaluation()
        self.assertNotEqual(first['policies']['always_flash_lite']['estimated_cost_usd'],
                            changed['policies']['always_flash_lite']['estimated_cost_usd'])
        self.assertEqual(self.grader.call_count, calls)

    def test_decisions_are_saved_before_private_decoding(self):
        """Hidden test outcomes cannot choose the fallback path."""
        original = benchmark.decode_private
        def decode(value):
            """Check decision ordering before delegating private decoding."""
            self.assertTrue((self.output/'decisions.json').exists())
            self.assertNotIn('PRIVATE_SECRET', (self.output/'decisions.json').read_text())
            return original(value)
        with patch('benchmark.decode_private', side_effect=decode) as decoder:
            self.run_evaluation()
        self.assertEqual(decoder.call_count, 2)
        self.assertTrue(all('PRIVATE_SECRET' not in p.read_text() for p in (self.output/'public').glob('*.json')))

    def test_missing_response_excludes_question_for_every_policy(self):
        """Compare the same questions for all policies when one API response is absent."""
        path = self.source/'records/q001-m02.json'
        record = load(path)
        record['status'] = 'request_error'
        del record['response']
        save(path, record)
        report = self.run_evaluation()
        self.assertEqual(report['coverage']['evaluated_questions'], 1)
        self.assertEqual(report['coverage']['source_received_responses'], 5)
        self.assertIn('fixture:1', report['coverage']['excluded_questions'])
        self.assertTrue(all(p['questions'] == 1 for p in report['policies'].values()))

    def test_truncated_responses_stay_in_comparison(self):
        """Do not discard incomplete answers to improve the measured model score."""
        path = self.source/'records/q001-m00.json'
        record = load(path)
        record['response']['choices'][0]['finish_reason'] = 'length'
        save(path, record)
        _, _, coverage = evaluator.load_inputs(self.source, self.questions)
        self.assertEqual(coverage['evaluated_questions'], 2)
        self.assertEqual(coverage['source_truncated_jobs'], ['q001-m00'])

    def test_changed_response_is_regraded(self):
        """Refresh only the changed answer's public and full verdicts."""
        self.run_evaluation()
        calls = self.grader.call_count
        path = self.source/'records/q000-m00.json'
        record = load(path)
        record['response']['choices'][0]['message']['content'] += '\n'
        save(path, record)
        self.run_evaluation()
        self.assertEqual(self.grader.call_count, calls + 2)

    def test_checker_changes_invalidate_verdict_cache(self):
        """A changed checker must not reuse prior grades."""
        self.run_evaluation()
        calls = self.grader.call_count
        with patch('assets.verify', return_value={'checker': 'changed'}):
            self.run_evaluation()
        self.assertEqual(self.grader.call_count, calls + 12)

    def test_unknown_verifier_is_retried_without_a_quality_verdict(self):
        """Infrastructure failures stop the run and can be retried with the same command."""
        with patch('benchmark.evaluate', return_value={'status': 'sandbox_error', 'success': None}):
            with self.assertRaisesRegex(RuntimeError, 'Checker unavailable'):
                self.run_evaluation()
        self.assertFalse((self.output/'decisions.json').exists())
        self.assertIsNone(load(self.output/'public/q000-m00.json')['success'])
        self.run_evaluation()
        self.assertEqual(load(self.output/'public/q000-m00.json')['success'], 1)

    def test_all_rejections_keep_last_answer_and_may_cost_more(self):
        """Fallback is not guaranteed to save money; retain negative savings."""
        def reject_public(response, sample):
            """Reject all public checks and use known hidden grades."""
            return fake_evaluate(response, sample) if 'PRIVATE_SECRET' in sample['input_output'] else {
                'status': 'scored', 'success': 0, 'verification_seconds': 0.25}
        with patch('benchmark.evaluate', side_effect=reject_public):
            report = self.run_evaluation()
        cascade = report['policies']['lite_then_flash_then_3_8']
        self.assertEqual(cascade['passed'], 2)
        self.assertEqual(cascade['fallbacks'], 2)
        self.assertLess(cascade['savings_percent'], 0)

    def test_invalid_model_identity_is_not_ignored(self):
        """A malformed received answer is an error rather than a missing response."""
        path = self.source/'records/q000-m00.json'
        record = load(path)
        record['response']['model'] = 'wrong-model'
        save(path, record)
        with self.assertRaisesRegex(ValueError, 'Invalid received'):
            self.run_evaluation()

    def test_source_run_cannot_be_output(self):
        """Protect the saved API responses from evaluator output writes."""
        with self.assertRaisesRegex(ValueError, 'separate'):
            evaluator.run(self.source, self.questions, self.source/'evaluation')

    def test_cli_needs_no_subcommand_or_configuration(self):
        """Calling evaluator.py alone must launch the fixed experiment."""
        with patch('sys.argv', ['evaluator.py']), patch('evaluator.run') as run:
            evaluator.main()
        run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
