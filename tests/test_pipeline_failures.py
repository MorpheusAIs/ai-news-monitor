import os
import unittest
from unittest.mock import patch

from api import grok, run


class PipelineFailureTests(unittest.TestCase):
    def test_ai_alpha_pipeline_fails_when_notion_publish_fails(self):
        with patch.dict(os.environ, {"NOTION_API_KEY": "notion-token"}, clear=False), \
            patch.object(run, "ai_alpha_page_exists", return_value=False), \
            patch.object(run, "fetch_reddit_posts", return_value=[{"title": "Launch", "ups": 5, "num_comments": 2}]), \
            patch.object(run, "filter_relevant_posts", return_value=[{"title": "Launch", "ups": 5, "num_comments": 2}]), \
            patch.object(run, "generate_summary", return_value="# AI Alpha - 2026-06-03\nSummary"), \
            patch.object(run, "publish_to_notion", side_effect=RuntimeError("notion rejected request")):
            with self.assertRaisesRegex(RuntimeError, "notion rejected request"):
                run.run_pipeline()

    def test_grok_pipeline_fails_when_notion_publish_fails(self):
        with patch.dict(os.environ, {"NOTION_API_KEY": "notion-token"}, clear=False), \
            patch.object(grok, "generate_grok_report", return_value="# Grok Alpha - 2026-06-03\nSummary"), \
            patch.object(grok, "publish_to_notion", side_effect=RuntimeError("notion rejected request")):
            with self.assertRaisesRegex(RuntimeError, "notion rejected request"):
                grok.run_grok_pipeline()

    def test_grok_report_uses_configurable_supported_model_chain(self):
        captured_payload = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "# Grok Alpha - 2026-06-03"}}]}

        class FakeClient:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, headers, json):
                captured_payload.update(json)
                return FakeResponse()

        env = {
            "OPENROUTER_API_KEY": "openrouter-token",
            "GROK_MODEL": "x-ai/grok-4.1",
            "GROK_FALLBACK_MODELS": "x-ai/grok-4.1-fast,x-ai/grok-4.1",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(grok.httpx, "Client", FakeClient):
            report = grok.generate_grok_report("2026-06-03")

        self.assertEqual(report, "# Grok Alpha - 2026-06-03")
        self.assertEqual(captured_payload["model"], "x-ai/grok-4.1")
        self.assertEqual(captured_payload["models"], ["x-ai/grok-4.1", "x-ai/grok-4.1-fast"])

    def test_ai_alpha_summary_rejects_null_openrouter_content(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": None}}]}

        class FakeClient:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, headers, json):
                return FakeResponse()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "openrouter-token"}, clear=False), \
            patch.object(run.httpx, "Client", FakeClient):
            with self.assertRaisesRegex(RuntimeError, "OpenRouter returned empty content"):
                run.generate_summary("posts", "2026-06-03")

    def test_grok_notion_publish_error_includes_response_body(self):
        class FakeHTTPStatusError(Exception):
            def __init__(self):
                super().__init__("fake notion http error")
                self.response = type("Response", (), {"status_code": 400, "text": "Could not find property Title"})()

        class FakeResponse:
            def raise_for_status(self):
                raise FakeHTTPStatusError()

        class FakeClient:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, headers, json):
                return FakeResponse()

        with patch.dict(os.environ, {"NOTION_API_KEY": "notion-token"}, clear=False), \
            patch.object(grok.httpx, "HTTPStatusError", FakeHTTPStatusError), \
            patch.object(grok.httpx, "Client", FakeClient):
            with self.assertRaisesRegex(RuntimeError, "Could not find property Title"):
                grok.grok_page_exists("notion-token", "database-id", "2026-06-03")


if __name__ == "__main__":
    unittest.main()
