# Step 2: Run models on fetched questions

This folder reads `data_fetcher/data/questions.json`, sends each question to
every configured model, and saves the full API responses and request status.
It uses Python 3.11 and the standard library on Linux or macOS; run
locking uses `fcntl`. It has no imports from the earlier pilot or router code.

## Models and settings

The default `models.json` calls Google AI Studio directly through Google's
Chat Completions compatibility endpoint. It contains:

- `gemini-3.1-flash-lite`
- `gemini-3-flash-preview` (Gemini 3 Flash)
- `gemini-3.8-flash`

All three use a 32,768-token completion cap, `high`
reasoning effort, and a 720-second HTTP timeout. Gemini 3 models use thinking
levels, so the configuration specifies effort instead of a fixed reasoning
token budget. These are collection settings, not measured results or proof of
equivalent reasoning across models.
Reasoning consumes part of the completion allowance. Model aliases and provider
defaults can change. Preparation checks model metadata; successful generation
with your account must still be verified.

Omitting `reasoning` uses the API's default. Choose settings supported by
the model. Each question is paired with every model, rotating model order
between questions. With 30 questions and three models, this makes 90 requests.

## Selecting an API

The configuration's `api` section selects the protocol, endpoint, and environment
variable containing the key. The default is:

```json
{
  "api": {
    "type": "chat_completions",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "api_key_env": "GEMINI_API_KEY"
  }
}
```

For another compatible API, change `base_url`, `api_key_env`, and model IDs.
The service must support bearer authentication, `GET /models/{id}`,
`POST /chat/completions`, `max_tokens`, and standard completion responses with
a matching model ID. If configured, it must also support `reasoning_effort`.
Native APIs with different request formats need another adapter in `backends.py`.

To use OpenRouter, pass `--config model_runner/models.openrouter.json` to
`preview` or `prepare`. That example selects `type: openrouter`, prefixed model
IDs, `OPENROUTER_API_KEY`, and explicit provider routing. OpenRouter additionally
supports a fixed `reasoning.max_tokens` budget; the generic backend accepts
only `reasoning.effort` and sends it as `reasoning_effort`.

One prepared run uses one API endpoint. All API settings are frozen in its
manifest. Older manifests without an `api` section retain OpenRouter behavior.

## Preview without network access

Run from the repository root after fetching the questions:

```bash
python3 model_runner/runner.py preview
```

Preview validates the inputs and prints the model list and request count. It
makes no network calls and creates no run directory. Use `--questions PATH` or
`--config PATH` to select another question file or configuration. If a sibling
`manifest.json` exists, its question checksum must match.

## Prepare a run

Set the key named by your configuration first. For the default AI Studio
configuration, enter it in Bash without recording its value in shell history:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
```

Then prepare the run:

```bash
python3 model_runner/runner.py prepare \
  --run-dir model_runner/runs/gemini-pilot
```

For AI Studio and other compatible APIs, export the configured key before
preparing: model metadata requests require authentication. OpenRouter metadata
requests are public. Neither backend generates responses during preparation.
Preparation freezes the prompts, settings, and model/provider information
in `manifest.json`, with a checksum in `manifest.sha256`. The directory must
be new.

## Collect responses (paid)

With the same API key still exported, collect the prepared requests:

```bash
python3 model_runner/runner.py collect \
  --run-dir model_runner/runs/gemini-pilot
```

The runner reads the environment only; it does not automatically load `.env`.
Collection uses the frozen run manifest, so subsequent edits to `models.json`
do not alter a prepared experiment. Only the problem statement and starter code
enter model requests. Dataset tests and metadata stay out of prompts.

Requests run sequentially with a lock for the whole run. Each request is
recorded before it is sent, and each response is saved before the next request.
Model identities are checked for both backends. OpenRouter additionally pins
the provider, disables fallbacks, checks provider parameter support during
preparation, and checks the returned provider identity. Generic model metadata
does not necessarily expose parameter support or token limits. A failed request
or unexpected identity stops collection.
Partial responses containing choice-level errors, filtered output, or unexpected
finish reasons also stop collection. HTTP failures retain a bounded error code,
type, and message with credentials redacted; raw headers and other provider
metadata are omitted.
No generated code is executed or graded in this step.

## Outputs and resuming

- `manifest.json` and `manifest.sha256`: frozen run inputs and provider metadata.
- `records/qNNN-mNN.json`: request, full API response, returned code text,
  finish reason, elapsed time, and status. Usage metadata is retained as
  returned by the API.
- `summary.json`: planned requests, attempted requests, and received responses.

Repeat the same `collect` command to resume. Received responses are skipped,
including responses truncated at the token cap. Truncation
is recorded in `finish_reason`; a received response does not mean a correct
solution. Interruptions before a confirmed response require manual review;
do not delete their records to force a retry. An unresolved record blocks
resumption because the original request may have been charged. Creating a
different run does not reuse previous paid responses.

## Tests and submission

```bash
python3 -m unittest discover -s model_runner -p 'test_*.py' -v
```

Tests use synthetic questions and mocked HTTP responses. Submit the source
files in this folder; generated runs and credentials are ignored.
`runner.py` owns validation, planning, persistence, and collection;
`backends.py` contains API-specific translation and metadata checks.
Use `runner.py` for both backends. Existing prepared OpenRouter runs can also
be collected with this entry point.

API references: [Google's Chat Completions endpoint](https://ai.google.dev/gemini-api/docs/openai),
[provider selection](https://openrouter.ai/docs/guides/routing/provider-selection),
[reasoning settings](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
and [usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
