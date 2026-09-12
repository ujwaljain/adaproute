"""Small JSON and checksum helpers shared by the evaluator and asset setup."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(raw):
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(raw).hexdigest()


def load(path):
    """Decode a UTF-8 JSON file; propagate malformed input and access errors."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Atomically replace a JSON file after flushing finite JSON to disk."""
    path = Path(path)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
