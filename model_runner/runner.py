"""Collect model API responses to fetched coding questions; never execute them."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import backends

HERE = Path(__file__).resolve().parent
BASE_URL = backends.OPENROUTER["base_url"]
SYSTEM = (
    "Solve the programming problem in Python 3.11. Return a single complete "
    "solution inside one ```python code block, with no explanatory text. "
    "Use only the Python standard library. Do not access files, networks, or external tools."
)


def digest(raw):
    """Return the SHA-256 hexadecimal digest of the supplied bytes."""
    return hashlib.sha256(raw).hexdigest()


def load(path):
    """Read UTF-8 JSON from a path and return its decoded Python value.

    File access errors and JSON decoding errors propagate to the caller.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Write a JSON value to a Path through a flushed temporary file.

    Replace the destination only after serialization and fsync succeed.
    The parent directory must exist. Non-finite JSON numbers are rejected;
    serialization and filesystem errors propagate to the caller.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def response_status(response, job):
    """Classify a response dictionary against a prepared job's identities.

    Return received, invalid_response, or unexpected_model_or_provider.
    A received response has text and finishes with stop or length. Reject
    top-level and choice-level errors, filtered output, and unexpected finish
    reasons. Correctness and completeness are not evaluated here.
    """
    if not isinstance(response, dict) or "error" in response:
        return "invalid_response"
    if response.get("model") != job["model"]:
        return "unexpected_model_or_provider"
    if job.get("expected_provider") is not None and response.get("provider") != job["expected_provider"]:
        return "unexpected_model_or_provider"
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "invalid_response"
    choice = choices[0]
    if "error" in choice or choice.get("finish_reason") not in ("stop", "length"):
        return "invalid_response"
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
        return "invalid_response"
    return "received"


def http_error_message(error, key=None, label="API"):
    """Return bounded HTTP failure details with credentials redacted.

    Read at most 16 KiB from an HTTPError and close its response. Include only
    the JSON error code, message, and metadata.error_type; omit headers and
    other provider metadata. Redact the active key and recognizable bearer
    tokens/API keys before limiting the message to 500 characters. Fall back
    to the HTTP status if the body cannot be read or parsed. Never retry.
    """
    details = []
    try:
        with error:
            payload = json.loads(error.read(16384))
        provider_error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(provider_error, dict):
            metadata = provider_error.get("metadata")
            fields = {"code": provider_error.get("code"),
                      "type": metadata.get("error_type") if isinstance(metadata, dict) else None,
                      "message": provider_error.get("message")}
            for name, value in fields.items():
                if type(value) in (str, int) and str(value).strip():
                    details.append(f"{name}={value}")
    except (OSError, HTTPException, ValueError):
        pass
    message = f"{label} HTTP {error.code}"
    if details:
        message += "; " + "; ".join(details)
    if key:
        message = message.replace(key, "[REDACTED]")
    message = re.sub(r"(?i)\bBearer\s+[^\s\"',;]+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    return " ".join(message.split())[:480] + "; no automatic retry"


def request_json(path, key=None, body=None, timeout=60, base_url=BASE_URL):
    """Make one API HTTP request and return its JSON response object.

    Args:
        path: API path relative to base_url, beginning with a slash.
        key: Optional bearer token, kept in the HTTP header only.
        body: JSON-serializable POST body, or None for a GET request.
        timeout: Socket-operation timeout in seconds.
        base_url: HTTPS endpoint chosen by the prepared configuration.

    Raises:
        RuntimeError: HTTP, transport, JSON decoding, or response-type failure.
            HTTP errors retain selected, redacted diagnostic fields; raw
            headers and provider metadata are omitted. No retry is attempted.
    """
    headers = {"Accept": "application/json", "User-Agent": "adaproute-model-runner/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    label = "OpenRouter" if base_url.rstrip("/") == BASE_URL else "API"
    request = Request(base_url.rstrip("/") + path, headers=headers,
                      data=None if body is None else json.dumps(body).encode())
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(http_error_message(exc, key, label)) from None
    except (OSError, HTTPException, ValueError):
        raise RuntimeError(f"{label} connection or response failed; no automatic retry") from None
    if not isinstance(result, dict):
        raise RuntimeError(f"{label} returned a non-object response")
    return result


def validate_questions(rows):
    """Validate the decoded question array without modifying its records.

    Require a nonempty list, unique platform/question_id pairs, nonempty
    problem statements, and textual starter code when supplied. Raise
    ValueError for invalid records; return None when all records are valid.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array of questions")
    seen_questions = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(field), str) or not row[field]
                                            for field in ("platform", "question_id")):
            raise ValueError("Each question needs a platform and question_id")
        key = f"{row['platform']}:{row['question_id']}"
        if key in seen_questions:
            raise ValueError("Duplicate question ID")
        seen_questions.add(key)
        statement, starter = row.get("question_content"), row.get("starter_code", "")
        if not isinstance(statement, str) or not statement.strip() or not isinstance(starter, str):
            raise ValueError("Question statement and starter code must be text")


def validate_model_config(config):
    """Validate decoded model and API configuration without network calls.

    Check generation limits, unique model IDs, provider slugs, and reasoning
    settings. Raise ValueError for invalid configuration; return None on
    success. Endpoint availability is checked separately by validate_providers().
    """
    required = {"max_tokens", "timeout_seconds", "models"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"api"}:
        raise ValueError("Config needs max_tokens, timeout_seconds, and models")
    backends.validate_api(config)
    for field in ("max_tokens", "timeout_seconds"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("Configure at least one model")
    seen_models = set()
    for model in models:
        if not isinstance(model, dict) or "id" not in model or set(model) - {"id", "provider", "reasoning"}:
            raise ValueError("Each model needs id and optional provider/reasoning")
        if not isinstance(model["id"], str) or not re.fullmatch(r"[\w.-]+(?:/[\w.:-]+)*", model["id"]):
            raise ValueError("Expected an explicit model ID")
        if model["id"] in seen_models:
            raise ValueError("Duplicate model ID")
        seen_models.add(model["id"])
        reasoning = model.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict) or set(reasoning) not in ({"max_tokens"}, {"effort"}):
                raise ValueError("Reasoning must specify either max_tokens or effort")
            if "max_tokens" in reasoning:
                tokens = reasoning["max_tokens"]
                if type(tokens) is not int or not 0 < tokens < config["max_tokens"]:
                    raise ValueError("Reasoning budget must be positive and below max_tokens")
            elif reasoning["effort"] not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                raise ValueError("Unsupported reasoning effort")
        backends.validate_model(config, model)


def validate_manifest(manifest, questions_sha256):
    """Validate the data fetcher's decoded manifest against question-file bytes.

    questions_sha256 is the expected hexadecimal digest of the question file.
    Require a JSON object containing a matching questions_sha256 field.
    Raise ValueError on mismatch and return None on success. Prepared-run
    manifest integrity is checked separately by read_manifest().
    """
    if not isinstance(manifest, dict) or manifest.get("questions_sha256") != questions_sha256:
        raise ValueError("Questions do not match the data fetcher's manifest")


def validate_inputs(questions_path, config_path):
    """Read and validate question, model, and optional source-manifest files.

    Args:
        questions_path: Path to the fetched JSON question array.
        config_path: Path to the JSON model configuration.

    Returns:
        A tuple of question records, model configuration, and question SHA-256.

    Raises:
        ValueError: Invalid JSON, questions, configuration, or manifest checksum.
        OSError: An input file cannot be read.

    A manifest.json beside the questions is validated when present. This
    function performs no network requests or file writes.
    """
    raw = Path(questions_path).read_bytes()
    rows, config = json.loads(raw), load(config_path)
    questions_sha256 = digest(raw)
    source_manifest = Path(questions_path).with_name("manifest.json")
    if source_manifest.exists():
        validate_manifest(load(source_manifest), questions_sha256)
    validate_questions(rows)
    validate_model_config(config)
    return rows, config, questions_sha256


def plan(questions_path, config_path):
    """Build a request plan from locally validated question/configuration files.

    Args:
        questions_path: Path to a JSON array produced by the data fetcher.
            A sibling manifest.json, when present, must match its checksum.
        config_path: Path to model IDs, provider slugs, and generation settings.

    Returns:
        A dictionary containing the source checksum, configuration, question
        count, and ordered jobs pairing every question with every model.

    Raises:
        ValueError: validate_inputs() rejects the input files.
        OSError: An input file cannot be read.

    This function neither uses the network nor writes files. Prompts contain
    only visible problem statements and starter code, never dataset tests.
    """
    rows, config, questions_sha256 = validate_inputs(questions_path, config_path)
    models = config["models"]
    jobs = []
    for index, row in enumerate(rows):
        key = f"{row['platform']}:{row['question_id']}"
        statement, starter = row["question_content"], row.get("starter_code", "")
        instruction = ("Complete this starter, preserving its class/function interface:\n"
                       f"```python\n{starter}\n```" if starter else
                       "Read input from standard input and write the answer to standard output. "
                       "Include the program entry point; do not hard-code the examples.")
        # Only these two visible fields enter prompts, never tests or metadata.
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": statement + "\n\n" + instruction}]
        for offset in range(len(models)):
            model_index = (index + offset) % len(models)
            model = models[model_index]
            body = backends.build_request(config, model, messages)
            jobs.append({"job_id": f"q{index:03d}-m{model_index:02d}",
                         "question_key": key, "model": model["id"], "request": body})
    return {"schema_version": 2, "questions_sha256": questions_sha256, "config": config,
            "question_count": len(rows), "jobs": jobs}


def validate_providers(config):
    """Check model metadata using the configured API backend.

    Args:
        config: Model configuration already checked by validate_model_config().

    Returns:
        A dictionary mapping each model ID to its verified provider metadata.

    Raises:
        ValueError: A provider is missing, ambiguous, or lacks required
            parameters, completion capacity, or identity information.
        RuntimeError: A metadata request fails.

    This function makes metadata GET requests, using authentication when
    required by the backend. It writes no files and never requests generation.
    """
    return backends.inspect_models(config, request_json)


def prepare(questions, config, run_dir):
    """Set up a run before sending questions to the models.

    Read the fetched questions and model configuration, check API metadata
    through the backend, and build one request for
    each question/model pair. Save those requests and settings in manifest.json,
    with a checksum in manifest.sha256, inside a new run directory.

    This step makes metadata requests, authenticated when the backend requires
    a key, without generation calls. collect() subsequently reads the saved
    manifest and sends its requests to the configured API.

    Local file validation is delegated to validate_inputs() through plan().
    Provider checks are delegated to validate_providers(); this function
    coordinates those steps and saves the prepared run.

    CLI example (from the repository root):
        python3 model_runner/runner.py prepare --run-dir model_runner/runs/gemini-pilot

    Args:
        questions: Path to the fetched JSON question array.
        config: Path to the model configuration JSON file.
        run_dir: New directory in which to save the manifest and records folder.

    Returns:
        The saved manifest, including requests and provider information.

    Raises:
        ValueError: Invalid inputs, existing output, or unsupported provider settings.
        RuntimeError: An API metadata request fails.
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise ValueError("Run directory exists; collect resumes it, prepare requires a new directory")
    manifest = plan(questions, config)
    endpoints = validate_providers(manifest["config"])
    for job in manifest["jobs"]:
        if backends.connection(manifest["config"])["type"] == "openrouter":
            job["expected_provider"] = endpoints[job["model"]]["provider_name"]
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), endpoints=endpoints)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "records").mkdir()
    save(run_dir / "manifest.json", manifest)
    (run_dir / "manifest.sha256").write_text(digest((run_dir / "manifest.json").read_bytes()) + "\n")
    return manifest


def read_manifest(run_dir):
    """Return a run directory's manifest after verifying its saved SHA-256.

    run_dir must be a Path. Raise ValueError if the bytes have changed since
    preparation; file access and JSON decoding errors propagate to the caller.
    """
    raw = (run_dir / "manifest.json").read_bytes()
    if digest(raw) != (run_dir / "manifest.sha256").read_text().strip():
        raise ValueError("Run manifest changed after preparation")
    return json.loads(raw)


@contextmanager
def collection_lock(run_dir):
    """Hold an exclusive advisory lock while collecting in a run directory.

    Use as a context manager with a run-directory Path. Raise ValueError if
    another collector holds the lock. Release it on context exit, retaining
    the lock file so concurrent processes continue to lock the same inode.
    """
    with (run_dir / ".collect.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another collector is using this run directory") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_records(run_dir, manifest):
    """Return existing job records in manifest order from a run-directory Path.

    Skip jobs without a saved record. Raise ValueError for mismatched job
    identities or requests, or invalid responses marked as received. This
    function reads records without modifying them or making network calls.
    """
    records = []
    for job in manifest["jobs"]:
        path = run_dir / "records" / (job["job_id"] + ".json")
        if path.exists():
            record = load(path)
            if any(record.get(field) != job[field] for field in ("job_id", "question_key", "model", "request")):
                raise ValueError("Saved record does not match the prepared job")
            if record.get("status") == "received" and response_status(record.get("response", {}), job) != "received":
                raise ValueError("Saved response is invalid or belongs to an unexpected model/provider")
            records.append(record)
    return records


def summary(run_dir, manifest):
    """Recompute, save, and return request and response counts.

    Read records from the run-directory Path using the supplied manifest.
    Counts cover attempted requests and successfully received responses.
    """
    records = read_records(run_dir, manifest)
    report = {"planned_requests": len(manifest["jobs"]), "attempted_requests": len(records),
              "received_responses": sum(record["status"] == "received" for record in records),
              "note": "Response collection only. No correctness evaluation."}
    save(run_dir / "summary.json", report)
    return report


def collect(run_dir):
    """Collect or resume a prepared run and return its latest summary.

    Args:
        run_dir: Path or path string naming an existing prepared run directory.

    Returns:
        A summary dictionary after all planned requests have been received.

    Raises:
        ValueError: A competing collector, changed manifest, unresolved prior
            request, missing API key, or failed response prevents further collection.
        OSError: Required local files cannot be read or persisted.

    Read the configured API key environment variable. Calls incur charges; save
    request intent before sending and each response before continuing. Skip
    confirmed responses and never automatically retry an
    unresolved request. Generated code is stored without being executed.
    """
    run_dir = Path(run_dir)
    with collection_lock(run_dir):
        manifest = read_manifest(run_dir)
        records = read_records(run_dir, manifest)
        report = summary(run_dir, manifest)
        if any(record["status"] != "received" for record in records):
            raise ValueError("Interrupted/failed request; review manually before resuming")
        completed = {record["job_id"] for record in records}
        remaining = [job for job in manifest["jobs"] if job["job_id"] not in completed]
        if not remaining:
            return report
        api = backends.connection(manifest["config"])
        key = backends.api_key(manifest["config"])
        for job in remaining:
            path = run_dir / "records" / (job["job_id"] + ".json")
            record = {**job, "status": "request_started",
                      "started_at": datetime.now(timezone.utc).isoformat()}
            # Persist intent before sending. A crash leaves a record that blocks retries.
            with path.open("x", encoding="utf-8") as output:
                json.dump(record, output)
                output.flush()
                os.fsync(output.fileno())
            started = time.monotonic()
            try:
                response = request_json("/chat/completions", key, job["request"],
                                        timeout=manifest["config"]["timeout_seconds"],
                                        base_url=api["base_url"])
                record.update(response=response, status=response_status(response, job))
                if record["status"] == "received":
                    choice = response["choices"][0]
                    record["finish_reason"] = choice.get("finish_reason")
                    record["content"] = choice["message"].get("content")
            except RuntimeError as exc:
                # request_json supplies a safe HTTP status or transport message.
                # Redact the active key defensively before persisting the error.
                record.update(status="request_error", error=str(exc).replace(key, "[REDACTED]")[:500])
            except (ValueError, OSError, HTTPException, KeyError, TypeError, IndexError):
                record.update(status="request_error", error="Request or response failed; no automatic retry")
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save(path, record)
            report = summary(run_dir, manifest)
            print(f"{job['job_id']} {job['model']}: {record['status']}", flush=True)
            if record["status"] != "received":
                raise ValueError("Request failed; saved the record and stopped without retries")
        return report


def main():
    """Parse command-line arguments and dispatch preview, prepare, or collect.

    Print request plans or collection summaries to standard output. Report
    handled failures to standard error and exit with status one. Only the
    collect subcommand sends paid generation requests.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--questions", type=Path, default=HERE.parent / "data_fetcher/data/questions.json")
        command.add_argument("--config", type=Path, default=HERE / "models.json")
        if name == "prepare":
            command.add_argument("--run-dir", type=Path, required=True)
    command = sub.add_parser("collect")
    command.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "collect":
            print(json.dumps(collect(args.run_dir), indent=2))
        else:
            result = (plan(args.questions, args.config) if args.command == "preview" else
                      prepare(args.questions, args.config, args.run_dir))
            print(f"{result['question_count']} questions / {len(result['jobs'])} requests")
            for model in result["config"]["models"]:
                print(f"  {model['id']}; reasoning={model.get('reasoning')}")
            print(f"API: {backends.connection(result['config'])['base_url']}")
            if args.command == "prepare":
                print(f"Prepared {args.run_dir}; no generation calls made.")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
