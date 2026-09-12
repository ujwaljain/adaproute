"""LiveCodeBench decoding, code extraction and isolated public/full test execution."""

import base64
import io
import json
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import tempfile
import time
import zlib

import assets


class DataOnlyUnpickler(pickle.Unpickler):
    """Decode the benchmark's pickled JSON string without executable globals."""

    def find_class(self, module, name):
        """Reject all global and class resolution during unpickling."""
        raise ValueError('Executable pickle globals are forbidden')

    def persistent_load(self, pid):
        """Reject external references during unpickling."""
        raise ValueError('Persistent pickle references are forbidden')


def decode_private(encoded):
    """Decode plain JSON or bounded base64/zlib data-only pickled JSON tests."""
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(encoded, validate=True), 64 * 1024**2)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('Private tests exceed decoding limit or are incomplete')
        value = DataOnlyUnpickler(io.BytesIO(raw)).load()
        if not isinstance(value, str):
            raise ValueError('Expected a pickled JSON string')
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError('Expected a list of private tests')
    return value


def sample(public_tests, function_name=None, private_tests=None):
    """Build a checker sample from explicit public tests and optional private tests.

    Public verification calls this without private_tests. Only final scoring
    supplies and decodes hidden tests, after fallback decisions are frozen.
    """
    tests = json.loads(public_tests)
    if not isinstance(tests, list) or not tests:
        raise ValueError('At least one public example is required')
    if private_tests is not None:
        tests += decode_private(private_tests)
    if any(not isinstance(t, dict) or not isinstance(t.get('input'), str)
           or not isinstance(t.get('output'), str) for t in tests):
        raise ValueError('Tests require textual input and output')
    if function_name is not None and (not isinstance(function_name, str) or not function_name.isidentifier()):
        raise ValueError('Invalid function interface')
    return {'input_output': json.dumps({'inputs': [t['input'] for t in tests],
                                       'outputs': [t['output'] for t in tests], 'fn_name': function_name})}


def extract_code(content):
    """Extract exactly one nonempty fenced Python block; reject ambiguous output."""
    if not isinstance(content, str):
        return None
    blocks = re.findall(r'```(?:python|py)?[ \t]*\r?\n(.*?)```', content, re.DOTALL | re.IGNORECASE)
    return blocks[0].strip() if len(blocks) == 1 and blocks[0].strip() else None


def command():
    """Build a Bubblewrap command with no host-code fallback or secret mounts."""
    if not shutil.which('bwrap'):
        raise RuntimeError('Bubblewrap is required; refusing host execution')
    return ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin',
            '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/runner',
            '--chdir', '/tmp', '--ro-bind', str(assets.CACHE/'testing_util.py'), '/runner/testing_util.py',
            '--ro-bind', str(Path(__file__).with_name('sandbox_worker.py')), '/runner/worker.py',
            '--ro-bind', str(assets.CACHE/'dependencies'), '/dependencies',
            '--clearenv', '--setenv', 'PATH', '/usr/bin',
            '--setenv', 'OPENBLAS_NUM_THREADS', '1',
            '--setenv', 'ADAPROUTE_EVALUATION_SANDBOX', '1',
            '/usr/bin/python3', '-I', '-B', '/runner/worker.py']


def grade_sample(test_sample, code, timeout=6):
    """Grade only inside the sandbox; infrastructure failures return unknown.

    The checker applies timeout per test. A 125-second wall cap and worker
    memory/CPU/output limits bound each solution. Resource/process failures
    outside normal checker verdicts are unknown, never silently wrong answers.
    """
    payload = json.dumps({'sample': test_sample, 'code': code, 'timeout': timeout}).encode()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(command(), input=payload, stdout=stdout, stderr=stderr,
                                    timeout=125, env={'PATH': '/usr/bin:/bin'})
        except subprocess.TimeoutExpired:
            return {'status': 'sandbox_wall_timeout', 'success': None}
        stdout.seek(0)
        stderr.seek(0)
        if result.returncode:
            return {'status': 'sandbox_error', 'success': None, 'returncode': result.returncode,
                    'error': stderr.read(2000).decode(errors='replace')}
        try:
            outcome = json.loads(stdout.read(8 * 1024**2))
            values = outcome['results']
            if not isinstance(values, list) or not values:
                raise ValueError('Empty or invalid checker results')
            if any(type(v) is not bool and not (type(v) is int and v in (-1, -2, -3, -4)) for v in values):
                raise ValueError('Unknown checker result values')
            passed = all(v is True for v in values)
            if passed and len(values) != len(json.loads(test_sample['input_output'])['inputs']):
                raise ValueError('Checker returned fewer passing verdicts than tests')
            return {'status': 'scored', 'success': int(passed),
                    'results': values, 'metadata': outcome.get('metadata')}
        except (ValueError, KeyError, TypeError) as error:
            return {'status': 'grader_protocol_error', 'success': None, 'error': str(error)}


def evaluate(response, test_sample, timeout=6):
    """Reject incomplete/ill-formatted output, otherwise grade its Python code."""
    started = time.monotonic()
    choice = response['choices'][0]
    code = extract_code(choice['message']['content'])
    if choice['finish_reason'] != 'stop':
        result = {'status': 'incomplete_response', 'success': 0}
    elif code is None:
        result = {'status': 'format_failure', 'success': 0}
    else:
        result = grade_sample(test_sample, code, timeout)
    return {**result, 'verification_seconds': round(time.monotonic() - started, 6)}


def selftest():
    """Verify the checker, negative verdict handling and OS isolation before use."""
    tests = sample('[{"input":"2\\n3\\n","output":"5\\n"}]')
    cases = [('a=int(input()); b=int(input()); print(a+b)', 1), ('print(0)', 0),
             ("raise ValueError('expected failure')", 0)]
    for code, expected in cases:
        result = grade_sample(tests, code)
        if result['success'] != expected:
            raise RuntimeError(f'Sandbox self-test failed: {result}')
    function = sample('[{"input":"[1,2,3]","output":"6"}]', 'total')
    if grade_sample(function, 'class Solution:\n    def total(self, nums): return sum(nums)')['success'] != 1:
        raise RuntimeError('Functional checker self-test failed')
    probe = ("import os, socket\nassert not os.path.exists('/home')\n"
             "assert not os.path.exists('/tmp/.env')\n"
             "assert not any('API_KEY' in key for key in os.environ)\n"
             "assert {name for _,name in socket.if_nameindex()} <= {'lo'}\n"
             "a=int(input()); b=int(input()); print(a+b)")
    if grade_sample(tests, probe)['success'] != 1:
        raise RuntimeError('Filesystem/environment/network isolation self-test failed')
