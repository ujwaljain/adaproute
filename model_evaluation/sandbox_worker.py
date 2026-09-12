"""Run only inside the filesystem and network isolation created by benchmark.py."""

import contextlib
import json
import os
import resource
import sys


def main():
    """Apply resource limits and invoke the pinned checker inside Bubblewrap."""
    if os.environ.get('ADAPROUTE_EVALUATION_SANDBOX') != '1' or os.path.exists('/home'):
        raise RuntimeError('This worker requires the isolated filesystem')
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path[:0] = ['/dependencies', '/runner']
    from testing_util import run_test
    task = json.load(sys.stdin)
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        results, metadata = run_test(task['sample'], test=task['code'],
                                     debug=False, timeout=task['timeout'])
    json.dump({'results': results, 'metadata': metadata}, output)
    output.write('\n')


if __name__ == '__main__':
    main()
