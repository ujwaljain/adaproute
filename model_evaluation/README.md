# Evaluate Gemini fallback savings

Run the saved questions through the public checker, select fallback answers,
then use hidden tests to compare their checker scores and estimated costs.
This folder makes no model API calls and has no imports from the other project folders.

## Run

The checker requires Linux x86_64, Python 3.11 and Bubblewrap (`bwrap`).
Install its pinned assets once:

```bash
python3 -B model_evaluation/assets.py
```

If the pilot cache already exists, reuse its downloaded files instead:

```bash
python3 -B model_evaluation/assets.py --from-cache .cache/livecodebench
```

Then run the whole evaluation:

```bash
python3 -B model_evaluation/evaluator.py
```

Defaults:

- Questions: `data_fetcher/data/questions.json`
- Responses: `model_runner/runs/gemini-30-questions`
- Results: `model_evaluation/runs/gemini-fallback-simple`

Optional `--questions`, `--source-run` and `--output-dir` arguments only change
these paths. There is no prepare step or configuration file. Run one evaluator
process at a time for a given output directory.

Run the same command again to reuse cached verdicts. A changed response, test
set or checker is re-graded automatically. Editing prices or fallback policies
only recalculates the comparison. Existing source responses and older evaluation
results are preserved; the simplified evaluator uses a separate output directory.

## This experiment

Model IDs and undiscounted rates are in `accounting.py`. Prices are USD per
million tokens, with thinking included in output:

| Model | Input | Output |
| --- | ---: | ---: |
| Gemini 3.1 Flash Lite | $0.25 | $1.50 |
| Gemini 3 Flash | $0.50 | $3.00 |
| Gemini 3.8 Flash | $1.50 | $7.50 |

Generated tokens are always `total_tokens - prompt_tokens` for these saved
Gemini responses. Their completion count can omit thinking tokens.
[Google token fields](https://ai.google.dev/api/generate-content#UsageMetadata).

`POLICIES` in `evaluator.py` contains the three models individually, Lite → 3.8,
Flash → 3.8, and Lite → Flash → 3.8. Always using 3.8 Flash is the cost baseline.
A cascade keeps the first answer passing public examples. If an answer is
truncated, has no single Python code block, or fails a public test, it tries the
next model. If all checks fail, it keeps the last answer and scores it as usual.
Decisions are saved before hidden tests are decoded or graded.

Every attempted model contributes to cost, including rejected answers:

```text
cost = (input tokens × input rate + generated tokens × output rate) / 1,000,000
savings % = 100 × (1 − total policy cost / total 3.8 Flash cost)
```

Missing API responses exclude the entire question from every policy, and the
report lists those exclusions. Truncated received answers stay in the comparison.
The current collection supplies 29 complete question/model sets out of 30.
These are offline generation estimates using the selected undiscounted rates;
already paid collection costs, HTTP retries, discounts and local checker compute
are not included. Passing public examples does not guarantee a correct solution.

## Checker and results

`benchmark.py` runs the pinned LiveCodeBench checker in Bubblewrap with a cleared
environment, no network and no repository or credentials mounted. Each solution
has a 1 GiB memory limit, a 120-second CPU limit and a 125-second wall limit;
the checker allows six seconds per test. Infrastructure failures stop the run
without assigning a quality verdict; fix the issue and rerun the same command.
The checker is intended for benchmark code, not deliberately malicious attempts
to forge the grader's result.

Scores mean passing this checker, whose rules differ from some original judges.
It rejects tiny float differences on `atcoder:abc392_d`, although the problem
allows a tolerance, and compares only one expected sequence on
`atcoder:abc396_e`, which allows multiple valid optimum sequences. These
questions remain in every policy's comparison; interpret scores accordingly.

`summary.json` contains checker scores, total costs, savings and fallback counts.
`policy-results.csv` gives per-question results. `decisions.json` records attempted
models by job ID; `public/` and `full/` contain cached verdicts. Runs and downloaded
checker assets are ignored by Git.

```bash
python3 -B -m unittest discover -s model_evaluation -p 'test_*.py'
```
