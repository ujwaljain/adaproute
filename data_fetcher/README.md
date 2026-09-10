# Step 1: Fetch 30 benchmark questions

This standalone submission downloads a pinned LiveCodeBench dataset and selects
the same 30 questions used in the original pilot: 10 easy, 10 medium, and 10 hard.
It uses only the Python standard library. Python 3.11 is the tested runtime.

## Run

From this folder:

```bash
python3 livecodebench.py
```

The download is approximately 134 MB. No API key, paid model calls, package
installation, or GPU is needed. The script writes:

- `data/questions.json`: a JSON array of the 30 original dataset records.
- `data/manifest.json`: source revision and checksum, sampling settings, ordered
  question IDs, and the checksum of `questions.json`.

The output directory must be new; existing output is preserved. Choose another
directory with `--output-dir data/another-sample` if needed.

To use an already downloaded copy without network access:

```bash
python3 livecodebench.py --source /path/to/test6.jsonl
```

The same pinned checksum is required for downloaded and local inputs.

## Selection

Source: `livecodebench/code_generation_lite`, revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, file `test6.jsonl`.
This incremental v6 slice contains 175 questions; it differs from the cumulative
`release_v6` dataset.

We use only `test6.jsonl` because the original pilot sampled its 30 questions
from this fixed pool, whose contest dates span January 4 through April 6, 2025.
Keeping that source and selection fixed reproduces the same questions for model
comparisons. This is a pilot scope choice, not a LiveCodeBench requirement:
the cumulative release includes earlier slices too. Sampling from that larger
pool would define a different experiment. The script downloads the entire
175-question slice before selecting 30; `test6` does not mean six questions.

1. Sort records by `platform:question_id`.
2. Remove duplicate IDs and duplicate statements after collapsing whitespace
   and lowercasing the text.
3. Within each difficulty, rank by SHA-256 of `42:<platform>:<question_id>`.
4. Take the first 10 per difficulty, ordered easy, medium, then hard.

These are previously analyzed pilot questions, not a fresh holdout. Matching the
original sample lets later steps compare models on the same questions.

All original fields are retained for later evaluation, including public and
private test cases. This step does not decode or execute tests. Future model
requests must use only the problem statement and starter code; never send the
whole record, which includes hidden tests.

## Submission contents and checks

Submit only this folder's four source files: `livecodebench.py`,
`test_livecodebench.py`, `README.md`, and `.gitignore`. The folder can be copied
outside the project and run independently. Downloaded questions and generated
manifests are local outputs, excluded by `.gitignore`.

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```

Tests use synthetic fixtures and mocked downloads. Later submissions can add
model execution, grading, and routing separately.
