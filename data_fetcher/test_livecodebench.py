import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import livecodebench as fetch


def fixture_rows():
    return [
        {"platform": "fixture", "question_id": f"{difficulty}-{i}",
         "difficulty": difficulty, "question_content": f"{difficulty} problem {i}",
         "private_test_cases": "opaque test data"}
        for difficulty in ("easy", "medium", "hard") for i in range(12)
    ]


class FetchQuestionsTests(unittest.TestCase):
    def test_selection_is_balanced_unique_and_stable(self):
        rows = fixture_rows()
        selected = fetch.select_questions(rows)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({fetch.question_key(row) for row in selected}), 30)
        for difficulty in ("easy", "medium", "hard"):
            self.assertEqual(sum(row["difficulty"] == difficulty for row in selected), 10)
        self.assertEqual(selected, fetch.select_questions(list(reversed(rows))))
        self.assertEqual(selected, fetch.select_questions(rows + rows))
        duplicate = dict(rows[0], question_id="zz-duplicate",
                         question_content="  EASY  PROBLEM  0\n")
        self.assertEqual(selected, fetch.select_questions(rows + [duplicate]))

    def test_insufficient_pool_fails(self):
        with self.assertRaisesRegex(ValueError, "Insufficient medium"):
            fetch.select_questions(fixture_rows()[:12])

    def test_download_writes_original_records_and_verifiable_manifest(self):
        rows = fixture_rows()
        raw = "\n".join(json.dumps(row) for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(fetch, "DATA_SHA256", fetch.digest(raw)), \
                patch.object(fetch.urllib.request, "urlopen", return_value=io.BytesIO(raw)) as download:
            root = Path(directory) / "sample"
            manifest = fetch.prepare(root)
            self.assertEqual(download.call_args.args[0].full_url, fetch.DATA_URL)
            questions = (root / "questions.json").read_bytes()
            selected = json.loads(questions)
            self.assertEqual(selected, fetch.select_questions(rows))
            self.assertEqual(manifest["questions_sha256"], fetch.digest(questions))
            self.assertEqual(manifest["question_keys"], [fetch.question_key(row) for row in selected])
            self.assertEqual(json.loads((root / "manifest.json").read_text()), manifest)
            self.assertEqual(manifest["source_question_count"], 36)

    def test_local_input_is_verified_without_network(self):
        raw = b'{"question_id": "fixture"}\n'
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(fetch, "DATA_SHA256", fetch.digest(raw)), \
                patch.object(fetch.urllib.request, "urlopen") as download:
            source = Path(directory) / "test6.jsonl"
            source.write_bytes(raw)
            self.assertEqual(fetch.load_questions(source), [{"question_id": "fixture"}])
            source.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                fetch.prepare(Path(directory) / "bad-output", source)
            self.assertFalse((Path(directory) / "bad-output").exists())
            download.assert_not_called()

    def test_bad_download_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(fetch.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            output = Path(directory) / "sample"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                fetch.prepare(output)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved_without_download(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(fetch.urllib.request, "urlopen") as download:
            original = Path(directory) / "questions.json"
            original.write_text("original")
            with self.assertRaises(FileExistsError):
                fetch.prepare(directory)
            self.assertEqual(original.read_text(), "original")
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
