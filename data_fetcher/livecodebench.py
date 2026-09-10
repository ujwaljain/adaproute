"""Fetch the reproducible 30-question LiveCodeBench pilot using only stdlib."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


DATA_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
DATA_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
    f"{DATA_REVISION}/test6.jsonl"
)
DATA_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
SEED = 42
PER_DIFFICULTY = 10
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def question_key(row):
    return f"{row['platform']}:{row['question_id']}"


def select_questions(rows):
    """Match the original pilot's deduplication, hash ranking, and ordering."""
    unique, seen_text = {}, set()
    for row in sorted(rows, key=question_key):
        normalized = " ".join(row["question_content"].split()).lower()
        fingerprint = digest(normalized.encode())
        if question_key(row) in unique or fingerprint in seen_text:
            continue
        unique[question_key(row)] = row
        seen_text.add(fingerprint)
    selected = []
    for difficulty in ("easy", "medium", "hard"):
        pool = [row for row in unique.values() if row["difficulty"] == difficulty]
        pool.sort(key=lambda row: digest(f"{SEED}:{question_key(row)}".encode()))
        if len(pool) < PER_DIFFICULTY:
            raise ValueError(f"Insufficient {difficulty} problems")
        selected.extend(pool[:PER_DIFFICULTY])
    return selected


def load_questions(source=None):
    """Verify the pinned JSONL bytes before parsing; never execute dataset code."""
    if source is not None:
        raw = Path(source).read_bytes()
    else:
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "adaproute-fetch-questions/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    if digest(raw) != DATA_SHA256:
        raise ValueError("Dataset SHA-256 mismatch; expected the pinned test6.jsonl")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def prepare(output_dir, source=None):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    rows = load_questions(source)
    selected = select_questions(rows)
    questions = (json.dumps(selected, indent=2) + "\n").encode("utf-8")
    manifest = {
        "dataset": "livecodebench/code_generation_lite",
        "dataset_revision": DATA_REVISION,
        "dataset_slice": "test6.jsonl (incremental v6)",
        "source_url": DATA_URL,
        "source_sha256": DATA_SHA256,
        "source_question_count": len(rows),
        "seed": SEED,
        "per_difficulty": PER_DIFFICULTY,
        "selection": "Deduplicate IDs and normalized statements; rank by SHA-256(seed:platform:id).",
        "question_count": len(selected),
        "question_keys": [question_key(row) for row in selected],
        "questions_sha256": digest(questions),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "questions.json").write_bytes(questions)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="New output directory (default: data/ beside this script)",
    )
    parser.add_argument(
        "--source", type=Path,
        help="Use an existing pinned test6.jsonl instead of downloading it",
    )
    args = parser.parse_args()
    try:
        manifest = prepare(args.output_dir, args.source)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {manifest['question_count']} questions: 10 easy, 10 medium, 10 hard.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
