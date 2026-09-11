"""Offline regression tests for model API response collection."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import runner


class RunnerTests(unittest.TestCase):
    """Exercise paired collection using temporary files and mocked HTTP calls."""

    def setUp(self):
        """Create isolated input fixtures and reject unexpected network access."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.questions = self.root / "questions.json"
        self.config = self.root / "models.json"
        self.run_dir = self.root / "run"
        self.rows = [
            {"platform": "fixture", "question_id": "1", "question_content": "Return the sum.",
             "starter_code": "class Solution: pass", "private_test_cases": "HIDDEN_TEST_SENTINEL",
             "metadata": "HIDDEN_METADATA_SENTINEL", "public_test_cases": "PUBLIC_TEST_SENTINEL"},
            {"platform": "fixture", "question_id": "2", "question_content": "Print the sum.",
             "starter_code": ""},
        ]
        self.settings = {"max_tokens": 64, "timeout_seconds": 120, "models": [
            {"id": "fixture/model-a", "provider": "fixture", "reasoning": {"max_tokens": 16}},
            {"id": "fixture/model-b", "provider": "fixture", "reasoning": {"effort": "low"}},
        ]}
        runner.save(self.questions, self.rows)
        runner.save(self.config, self.settings)
        self.network = patch.object(runner, "urlopen", side_effect=AssertionError("Unexpected network access"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def endpoint(self, path, **kwargs):
        """Validate a public metadata GET and return a supported fixture endpoint."""
        self.assertTrue(path.endswith("/endpoints"))
        self.assertNotIn("key", kwargs)
        return {"data": {"endpoints": [{
            "tag": "fixture", "provider_name": "Fixture Provider",
            "supported_parameters": ["max_tokens", "reasoning"], "max_completion_tokens": 1000,
        }]}}

    def prepare(self):
        """Prepare the fixture run with mocked endpoint metadata."""
        with patch.object(runner, "request_json", side_effect=self.endpoint):
            return runner.prepare(self.questions, self.config, self.run_dir)

    def response(self, path, key=None, body=None, timeout=None, base_url=None):
        """Check a generation request and return a matching response with usage."""
        self.assertEqual(path, "/chat/completions")
        self.assertEqual(key, "fixture-key")
        self.assertEqual(timeout, 120)
        self.assertEqual(base_url, runner.BASE_URL)
        self.assertNotIn("SENTINEL", json.dumps(body))
        return {"id": "fixture-generation", "model": body["model"], "provider": "Fixture Provider",
                "choices": [{"finish_reason": "stop", "message": {"content": "```python\nprint(1)\n```"}}],
                "usage": {"cost": "0.01", "is_byok": False, "prompt_tokens": 20,
                          "completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 4}}}

    def collect(self, response=None):
        """Collect fixtures and return the summary plus the mock HTTP callable.

        response optionally supplies a replacement callback for failure cases.
        Collection errors propagate so tests can assert the stopping behavior.
        """
        with patch.dict(runner.os.environ, {"OPENROUTER_API_KEY": "fixture-key"}), \
                patch.object(runner, "request_json", side_effect=response or self.response) as request, \
                redirect_stdout(io.StringIO()):
            report = runner.collect(self.run_dir)
        return report, request

    def test_preview_is_offline_and_pairs_every_question_with_every_model(self):
        """Verify complete pairing, rotated order, and exclusion of dataset tests."""
        plan = runner.plan(self.questions, self.config)
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual([job["model"] for job in plan["jobs"]],
                         ["fixture/model-a", "fixture/model-b", "fixture/model-b", "fixture/model-a"])
        first, last = plan["jobs"][0]["request"], plan["jobs"][-1]["request"]
        self.assertIn("class Solution", first["messages"][1]["content"])
        self.assertIn("standard input", last["messages"][1]["content"])
        self.assertNotIn("SENTINEL", json.dumps(plan))
        self.assertFalse(self.run_dir.exists())

    def test_source_manifest_mismatch_and_duplicate_questions_are_rejected(self):
        """Reject changed source bytes and repeated question identities."""
        path = self.root / "manifest.json"
        runner.save(path, {"questions_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "manifest"):
            runner.plan(self.questions, self.config)
        runner.save(self.questions, self.rows + self.rows)
        runner.save(path, {"questions_sha256": runner.digest(self.questions.read_bytes())})
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            runner.plan(self.questions, self.config)

    def test_invalid_configuration_fails_before_any_network(self):
        """Reject invalid limits, empty/duplicate models, and oversized reasoning."""
        changes = [{"max_tokens": 0}, {"timeout_seconds": True}, {"models": []},
                   {"models": self.settings["models"] * 2},
                   {"models": [{"id": "fixture/a", "provider": "fixture", "reasoning": {"max_tokens": 64}}]}]
        for change in changes:
            with self.subTest(change=change):
                runner.save(self.config, {**self.settings, **change})
                with self.assertRaises(ValueError):
                    runner.plan(self.questions, self.config)

    def test_invalid_question_records_stop_preparation_before_network(self):
        """Reject missing fields and invalid question types before provider calls."""
        invalid_inputs = [[], {}, [None], [{"platform": "fixture", "question_id": "1"}],
                          [{**self.rows[0], "question_id": 1}],
                          [{**self.rows[0], "question_content": "   "}],
                          [{**self.rows[0], "starter_code": None}]]
        for rows in invalid_inputs:
            with self.subTest(rows=rows):
                runner.save(self.questions, rows)
                with self.assertRaises(ValueError):
                    runner.prepare(self.questions, self.config, self.run_dir)
                self.assertFalse(self.run_dir.exists())

    def test_invalid_source_manifest_stops_preparation_before_network(self):
        """Reject non-object manifests and missing checksums before provider calls."""
        for manifest in (None, [], "invalid", {}):
            with self.subTest(manifest=manifest):
                runner.save(self.root / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "manifest"):
                    runner.prepare(self.questions, self.config, self.run_dir)
                self.assertFalse(self.run_dir.exists())

    def test_prepare_fetches_metadata_only_and_configures_provider(self):
        """Check frozen manifests and provider guards without generation calls."""
        manifest = self.prepare()
        self.assertEqual(runner.read_manifest(self.run_dir), manifest)
        provider = manifest["jobs"][0]["request"]["provider"]
        self.assertTrue(provider["require_parameters"])
        self.assertFalse(provider["allow_fallbacks"])
        self.assertEqual(provider["only"], ["fixture"])
        self.assertEqual(list((self.run_dir / "records").iterdir()), [])
        self.assertNotIn("SENTINEL", (self.run_dir / "manifest.json").read_text())

    def test_unsupported_provider_stops_before_generation(self):
        """Reject endpoints lacking required generation parameters."""
        unavailable = self.endpoint("/endpoints")
        unavailable["data"]["endpoints"][0]["supported_parameters"] = ["max_tokens"]
        with patch.object(runner, "request_json", return_value=unavailable), \
                self.assertRaisesRegex(ValueError, "parameters"):
            runner.prepare(self.questions, self.config, self.run_dir)
        self.assertFalse(self.run_dir.exists())

    def test_collection_saves_each_response_and_resume_does_not_repeat_calls(self):
        """Verify response persistence and completion-aware resuming."""
        self.prepare()
        report, request = self.collect()
        self.assertEqual(request.call_count, 4)
        self.assertEqual(report["received_responses"], 4)
        self.assertEqual(report["attempted_requests"], 4)
        record = runner.load(self.run_dir / "records/q000-m00.json")
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["response"]["usage"]["completion_tokens_details"]["reasoning_tokens"], 4)
        self.assertNotIn("fixture-key", json.dumps(record))
        report2, request2 = self.collect()
        self.assertEqual(report2, report)
        request2.assert_not_called()

    def test_started_request_is_saved_before_network_and_blocks_retries(self):
        """Simulate an interrupted paid request and verify it cannot be resent."""
        self.prepare()

        def interrupted(*args, **kwargs):
            """Assert persisted request intent, then simulate process interruption."""
            record = runner.load(self.run_dir / "records/q000-m00.json")
            self.assertEqual(record["status"], "request_started")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.collect(interrupted)
        with patch.object(runner, "request_json") as request, \
                self.assertRaisesRegex(ValueError, "Interrupted"):
            runner.collect(self.run_dir)
        request.assert_not_called()
        report = runner.load(self.run_dir / "summary.json")
        self.assertEqual(report["attempted_requests"], 1)
        self.assertEqual(report["received_responses"], 0)

    def test_collection_and_resume_work_without_usage_or_cost_fields(self):
        """Collect and resume valid responses without requiring billing metadata."""
        self.prepare()

        def without_usage(*args, **kwargs):
            """Return a valid fixture response without usage metadata."""
            response = self.response(*args, **kwargs)
            del response["usage"]
            return response

        report, request = self.collect(without_usage)
        self.assertEqual(request.call_count, 4)
        self.assertEqual(report["received_responses"], 4)
        report2, request2 = self.collect()
        self.assertEqual(report2, report)
        request2.assert_not_called()

    def test_failed_request_preserves_http_status_and_redacts_key(self):
        """Save actionable failure details without exposing the active API key."""
        self.prepare()
        with patch.dict(runner.os.environ, {"OPENROUTER_API_KEY": "fixture-key"}), \
                patch.object(runner, "request_json", side_effect=RuntimeError(
                    "OpenRouter HTTP 401; fixture-key; no automatic retry")) as request, \
                redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "stopped without retries"):
            runner.collect(self.run_dir)
        self.assertEqual(request.call_count, 1)
        record = runner.load(self.run_dir / "records/q000-m00.json")
        self.assertIn("HTTP 401", record["error"])
        self.assertNotIn("fixture-key", json.dumps(record))
        self.assertEqual(record["status"], "request_error")

    def test_wrong_model_or_malformed_response_stops_and_preserves_response(self):
        """Preserve unexpected responses while stopping further paid requests."""
        for modification, status in [({"model": "fixture/unrequested"}, "unexpected_model_or_provider"),
                                     ({"choices": [None]}, "invalid_response")]:
            with self.subTest(status=status):
                self.run_dir = self.root / status
                self.prepare()

                def bad(*args, **kwargs):
                    """Alter the response identity or structure for this subtest."""
                    return {**self.response(*args, **kwargs), **modification}

                with self.assertRaisesRegex(ValueError, "stopped without retries"):
                    self.collect(bad)
                record = runner.load(self.run_dir / "records/q000-m00.json")
                self.assertEqual(record["status"], status)
                self.assertIn("response", record)

    def test_manifest_changes_block_requests(self):
        """Reject a manifest modified after preparation."""
        manifest = self.prepare()
        manifest["config"]["max_tokens"] = 32
        runner.save(self.run_dir / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.collect()

    def test_failed_choices_stop_collection_and_block_resume(self):
        """Reject partial failures even when the HTTP response contains text."""
        changes = [{"error": {"code": 502, "message": "Provider failed"}},
                   {"finish_reason": "error"}, {"finish_reason": "content_filter"},
                   {"finish_reason": "tool_calls"}, {"finish_reason": ""},
                   {"finish_reason": None}, {"finish_reason": "unknown"}]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                self.run_dir = self.root / f"failed-choice-{index}"
                self.prepare()

                def failed_choice(*args, **kwargs):
                    """Return partial output with the failure under test."""
                    response = self.response(*args, **kwargs)
                    response["choices"][0].update(change)
                    return response

                with self.assertRaisesRegex(ValueError, "stopped without retries"):
                    self.collect(failed_choice)
                records = list((self.run_dir / "records").glob("*.json"))
                self.assertEqual(len(records), 1)
                record = runner.load(records[0])
                self.assertEqual(record["status"], "invalid_response")
                self.assertEqual(record["response"]["choices"][0]["message"]["content"],
                                 "```python\nprint(1)\n```")
                self.assertEqual(runner.load(self.run_dir / "summary.json")["received_responses"], 0)
                with patch.object(runner, "request_json") as request, \
                        self.assertRaisesRegex(ValueError, "Interrupted/failed"):
                    runner.collect(self.run_dir)
                request.assert_not_called()

    def test_truncated_responses_are_preserved_without_repeat_calls(self):
        """Keep length-limited responses resumable without calling them complete."""
        self.prepare()

        def truncated(*args, **kwargs):
            """Return text that stops at the output token limit."""
            response = self.response(*args, **kwargs)
            response["choices"][0]["finish_reason"] = "length"
            return response

        report, request = self.collect(truncated)
        self.assertEqual(report["received_responses"], 4)
        record = runner.load(self.run_dir / "records/q000-m00.json")
        self.assertEqual(record["finish_reason"], "length")
        report2, request2 = self.collect()
        self.assertEqual(report, report2)
        request2.assert_not_called()

    def test_http_diagnostics_are_saved_with_secrets_and_metadata_removed(self):
        """Retain actionable provider errors without retaining credentials."""
        self.prepare()
        payload = {"error": {"code": 404,
                   "message": "No endpoints for fixture-key; Bearer another-secret sk-or-v1-secret",
                   "metadata": {"error_type": "provider_unavailable", "raw": "RAW_SENTINEL",
                                "headers": {"Authorization": "HEADER_SENTINEL"}}}}
        body = io.BytesIO(json.dumps(payload).encode())
        error = HTTPError(runner.BASE_URL, 404, "HEADER_SENTINEL", {}, body)
        with patch.dict(runner.os.environ, {"OPENROUTER_API_KEY": "fixture-key"}), \
                patch.object(runner, "urlopen", side_effect=error) as network, \
                redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "stopped without retries"):
            runner.collect(self.run_dir)
        self.assertEqual(network.call_count, 1)
        self.assertTrue(body.closed)
        record = runner.load(self.run_dir / "records/q000-m00.json")
        self.assertEqual(record["status"], "request_error")
        for expected in ("HTTP 404", "code=404", "provider_unavailable", "No endpoints", "[REDACTED]"):
            self.assertIn(expected, record["error"])
        for secret in ("fixture-key", "another-secret", "sk-or-v1-secret", "RAW_SENTINEL", "HEADER_SENTINEL"):
            self.assertNotIn(secret, json.dumps(record))
        self.assertEqual(runner.load(self.run_dir / "summary.json")["received_responses"], 0)

    def test_http_diagnostics_handle_unusable_bodies_and_limit_message_size(self):
        """Handle non-JSON error pages and oversized payloads without echoing them."""
        for raw in (b"<html>secret</html>", b"[]", b'{"error": "secret"}',
                    b'{"error": null}', b"x" * 20000):
            with self.subTest(raw_length=len(raw)):
                error = HTTPError(runner.BASE_URL, 502, "secret", {}, io.BytesIO(raw))
                with patch.object(runner, "urlopen", side_effect=error), \
                        self.assertRaises(RuntimeError) as caught:
                    runner.request_json("/chat/completions", "fixture-key", {})
                self.assertEqual(str(caught.exception), "OpenRouter HTTP 502; no automatic retry")
        payload = {"error": {"message": "x" * 460 + "fixture-key" + "y" * 1000}}
        error = HTTPError(runner.BASE_URL, 400, "", {}, io.BytesIO(json.dumps(payload).encode()))
        message = runner.http_error_message(error, "fixture-key")
        self.assertLessEqual(len(message), 500)
        self.assertNotIn("fixture-key", message)
        self.assertNotIn("fixture-", message)

    def test_run_lock_prevents_a_second_collector(self):
        """Verify concurrent collectors cannot share the same run directory."""
        self.prepare()
        with runner.collection_lock(self.run_dir), \
                self.assertRaisesRegex(ValueError, "Another collector"):
            self.collect()

    def test_http_request_uses_bearer_auth_and_does_not_echo_error_payload(self):
        """Check request authentication and prevent error messages exposing keys."""
        with patch.object(runner, "urlopen", return_value=io.BytesIO(b'{"ok": true}')) as network:
            self.assertEqual(runner.request_json("/chat/completions", "fixture-key", {"model": "fixture/a"}),
                             {"ok": True})
            request = network.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer fixture-key")
            self.assertEqual(json.loads(request.data), {"model": "fixture/a"})
        error = HTTPError(runner.BASE_URL, 401, "fixture-key", {}, io.BytesIO(b'fixture-key'))
        with patch.object(runner, "urlopen", side_effect=error), \
                self.assertRaises(RuntimeError) as caught:
            runner.request_json("/chat/completions", "fixture-key", {})
        self.assertNotIn("fixture-key", str(caught.exception))

    def direct_settings(self, base_url="https://generativelanguage.googleapis.com/v1beta/openai"):
        """Build a direct API configuration using native Gemini model IDs."""
        return {"api": {"type": "chat_completions", "base_url": base_url,
                        "api_key_env": "GEMINI_API_KEY"},
                "max_tokens": 64, "timeout_seconds": 120,
                "models": [{"id": "gemini-3.1-flash-lite", "reasoning": {"effort": "high"}},
                           {"id": "gemini-3-flash-preview", "reasoning": {"effort": "high"}}]}

    def test_direct_api_preview_is_offline_and_uses_native_parameters(self):
        """Translate shared reasoning settings without leaking OpenRouter routing."""
        runner.save(self.config, self.direct_settings())
        plan = runner.plan(self.questions, self.config)
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual(plan["schema_version"], 2)
        for job in plan["jobs"]:
            self.assertEqual(job["request"]["reasoning_effort"], "high")
            self.assertEqual(job["request"]["max_tokens"], 64)
            self.assertNotIn("reasoning", job["request"])
            self.assertNotIn("provider", job["request"])
        self.assertNotIn("SENTINEL", json.dumps(plan))
        self.assertFalse(self.run_dir.exists())

    def test_direct_api_transport_prepare_collect_and_resume(self):
        """Exercise authenticated metadata and generation through two API URLs."""
        for index, url in enumerate(("https://generativelanguage.googleapis.com/v1beta/openai",
                                     "https://compatible.example/v1/")):
            with self.subTest(url=url):
                config = self.direct_settings(url)
                self.run_dir = self.root / f"direct-{index}"
                runner.save(self.config, config)

                def respond(request, timeout):
                    """Check the actual HTTP request and emulate a compatible API."""
                    self.assertTrue(request.full_url.startswith(url.rstrip("/") + "/"))
                    self.assertEqual(request.get_header("Authorization"), "Bearer direct-key")
                    self.assertNotIn("direct-key", request.full_url)
                    if request.get_method() == "GET":
                        self.assertIn("/models/", request.full_url)
                        model_id = request.full_url.rsplit("/", 1)[1]
                        prefix = "models/" if "generativelanguage.googleapis.com" in url else ""
                        result = {"id": prefix + model_id, "object": "model"}
                    else:
                        self.assertEqual(request.full_url, url.rstrip("/") + "/chat/completions")
                        self.assertEqual(timeout, 120)
                        body = json.loads(request.data)
                        self.assertNotIn("provider", body)
                        self.assertNotIn("reasoning", body)
                        self.assertEqual(body["reasoning_effort"], "high")
                        self.assertNotIn("SENTINEL", str(body))
                        result = {"model": body["model"], "choices": [
                            {"finish_reason": "stop", "message": {"content": "```python\npass\n```"}}]}
                    return io.BytesIO(json.dumps(result).encode())

                with patch.dict(runner.os.environ, {"GEMINI_API_KEY": "direct-key",
                                                     "OPENROUTER_API_KEY": "wrong-key"}), \
                        patch.object(runner, "urlopen", side_effect=respond) as network, \
                        redirect_stdout(io.StringIO()):
                    manifest = runner.prepare(self.questions, self.config, self.run_dir)
                    self.assertEqual(network.call_count, 2)
                    self.assertNotIn("expected_provider", manifest["jobs"][0])
                    config["api"]["base_url"] = "https://changed.example/v1"
                    runner.save(self.config, config)
                    report = runner.collect(self.run_dir)
                    self.assertEqual(network.call_count, 6)
                    self.assertEqual(report["received_responses"], 4)
                    report2 = runner.collect(self.run_dir)
                    self.assertEqual(network.call_count, 6)
                    self.assertEqual(report, report2)
                for path in self.run_dir.rglob("*.json"):
                    self.assertNotIn("direct-key", path.read_text())
                    self.assertNotIn("wrong-key", path.read_text())

    def test_direct_api_requires_selected_key_before_metadata_or_intent(self):
        """Do not substitute an OpenRouter key or record a request without a key."""
        config = self.direct_settings()
        runner.save(self.config, config)
        with patch.dict(runner.os.environ, {"GEMINI_API_KEY": "", "OPENROUTER_API_KEY": "other"}), \
                self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            runner.prepare(self.questions, self.config, self.run_dir)
        self.assertFalse(self.run_dir.exists())
        manifest = runner.plan(self.questions, self.config)
        self.run_dir.mkdir()
        (self.run_dir / "records").mkdir()
        runner.save(self.run_dir / "manifest.json", manifest)
        (self.run_dir / "manifest.sha256").write_text(runner.digest((self.run_dir / "manifest.json").read_bytes()))
        with patch.dict(runner.os.environ, {"GEMINI_API_KEY": ""}), \
                self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            runner.collect(self.run_dir)
        self.assertEqual(list((self.run_dir / "records").iterdir()), [])

    def test_direct_api_rejects_wrong_model_metadata_and_response(self):
        """Retain model identity checks even when no provider field is returned."""
        runner.save(self.config, self.direct_settings())
        with patch.dict(runner.os.environ, {"GEMINI_API_KEY": "direct-key"}), \
                patch.object(runner, "request_json", return_value={"id": "wrong-model"}), \
                self.assertRaisesRegex(ValueError, "identity mismatch"):
            runner.prepare(self.questions, self.config, self.run_dir)
        self.assertFalse(self.run_dir.exists())
        job = runner.plan(self.questions, self.config)["jobs"][0]
        response = {"model": "wrong-model", "choices": [
            {"finish_reason": "stop", "message": {"content": "partial"}}]}
        self.assertEqual(runner.response_status(response, job), "unexpected_model_or_provider")
        response["model"] = job["model"]
        response["choices"][0]["error"] = {"message": "failed"}
        self.assertEqual(runner.response_status(response, job), "invalid_response")

    def test_invalid_api_configuration_and_protocol_settings_are_rejected(self):
        """Reject invalid endpoints, secret-bearing URLs, and protocol mismatches."""
        changes = [{"type": "unknown"}, {"base_url": "http://example.com/v1"},
                   {"base_url": "https://user:key@example.com/v1"},
                   {"base_url": "https://example.com/v1?key=secret"},
                   {"base_url": "https://example.com/v1#fragment"},
                   {"api_key_env": "sk-real-key"}, {"type": "openrouter"}]
        for change in changes:
            with self.subTest(change=change):
                config = self.direct_settings()
                config["api"].update(change)
                runner.save(self.config, config)
                with self.assertRaises(ValueError):
                    runner.prepare(self.questions, self.config, self.run_dir)
        for change in ({"provider": "google-ai-studio"}, {"reasoning": {"max_tokens": 16}}):
            config = self.direct_settings()
            config["models"][0].update(change)
            with self.assertRaises(ValueError):
                runner.validate_model_config(config)

    def test_direct_api_http_errors_are_redacted_and_named_generically(self):
        """Report direct API failures without labeling them as OpenRouter errors."""
        payload = {"error": {"code": 403, "message": "Invalid direct-key"}}
        error = HTTPError("https://compatible.example/v1", 403, "", {},
                          io.BytesIO(json.dumps(payload).encode()))
        with patch.object(runner, "urlopen", side_effect=error), \
                self.assertRaises(RuntimeError) as caught:
            runner.request_json("/chat/completions", "direct-key", {},
                                base_url="https://compatible.example/v1")
        self.assertIn("API HTTP 403", str(caught.exception))
        self.assertNotIn("OpenRouter", str(caught.exception))
        self.assertNotIn("direct-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
