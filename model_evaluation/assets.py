"""Install and verify the pinned checker and NumPy wheel; never call model APIs."""

import argparse
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import zipfile

from common import digest, save

HERE = Path(__file__).resolve().parent
CACHE = HERE / '.cache'
REVISION = '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24'
CODE_BASE = f'https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/{REVISION}'
WHEEL = 'numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
SOURCES = {
    'testing_util.py': (CODE_BASE + '/lcb_runner/evaluation/testing_util.py',
                        'b7cb6a8a69807bb868150a61742e25d7bb5328bbe01d514471b6ec43c9fa9ed2'),
    'LICENSE': (CODE_BASE + '/LICENSE',
                '104099948b78ebc36f2e951b3a7973b956adfa7d39aaec0d1512f97a2686d15d'),
    WHEEL: ('https://files.pythonhosted.org/packages/b3/dd/2238b898e51bd6d389b7389ffb20d7f4c10066d80351187ec8e303a5a475/' + WHEEL,
            'ba10f8411898fc418a521833e014a77d3ca01c15b0c6cdcce6a0d2897e6dbbdf'),
}


def expected_dependencies():
    """Derive dependency hashes from the verified wheel, rejecting unsafe entries."""
    expected = {}
    with zipfile.ZipFile(CACHE / WHEEL) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if (path.is_absolute() or '..' in path.parts
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe wheel member')
            if not member.is_dir():
                if member.filename in expected:
                    raise ValueError('Duplicate wheel member')
                expected[member.filename] = digest(archive.read(member))
    return expected


def verify():
    """Verify pinned downloads and every extracted dependency before execution."""
    for name, (_, checksum) in SOURCES.items():
        path = CACHE / name
        if not path.is_file() or path.is_symlink() or digest(path.read_bytes()) != checksum:
            raise ValueError(f'Missing or changed asset {name}; run model_evaluation/assets.py')
    expected = expected_dependencies()
    dependencies = CACHE / 'dependencies'
    paths = list(dependencies.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Dependency symlinks are not allowed')
    actual = {str(p.relative_to(dependencies)): digest(p.read_bytes()) for p in paths if p.is_file()}
    if actual != expected:
        raise ValueError('Extracted dependencies differ from the pinned wheel')
    return {'checker_revision': REVISION,
            'files': {name: checksum for name, (_, checksum) in SOURCES.items()},
            'dependencies_sha256': digest(json.dumps(expected, sort_keys=True).encode())}


def setup(from_cache=None):
    """Copy or download hash-pinned assets and extract the wheel without pip.

    from_cache optionally supplies already downloaded files; their original
    directory is never modified. The resulting folder has no pilot dependency.
    Network downloads are limited to the three explicit public URLs above.
    """
    CACHE.mkdir(exist_ok=True)
    for name, (url, checksum) in SOURCES.items():
        target = CACHE / name
        if not target.exists():
            if from_cache is not None:
                raw = (Path(from_cache) / name).read_bytes()
            else:
                print('Downloading', name, flush=True)
                with urlopen(Request(url, headers={'User-Agent': 'adaproute-evaluator/1.0'}), timeout=60) as response:
                    raw = response.read(32 * 1024**2)
            if digest(raw) != checksum:
                raise ValueError(f'Asset checksum mismatch: {name}')
            target.write_bytes(raw)
        if target.is_symlink() or digest(target.read_bytes()) != checksum:
            raise ValueError(f'Asset checksum mismatch: {name}')
    expected_dependencies()
    if not (CACHE / 'dependencies').exists():
        with tempfile.TemporaryDirectory(dir=CACHE) as temporary:
            extracted = Path(temporary) / 'dependencies'
            extracted.mkdir()
            with zipfile.ZipFile(CACHE / WHEEL) as archive:
                archive.extractall(extracted)
            extracted.rename(CACHE / 'dependencies')
    fingerprint = verify()
    save(CACHE / 'assets.json', fingerprint)
    print('Verified standalone checker assets:', CACHE)


def main():
    """Parse optional cache reuse and prepare local checker assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache', type=Path)
    args = parser.parse_args()
    try:
        setup(args.from_cache)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
