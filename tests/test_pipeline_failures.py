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
            "GROK_MODEL": "x-ai/grok-4.3",
            "GROK_FALLBACK_MODELS": "x-ai/grok-4.20,x-ai/grok-4.3",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(grok.httpx, "Client", FakeClient):
            report = grok.generate_grok_report("2026-06-03")

        self.assertEqual(report, "# Grok Alpha - 2026-06-03")
        self.assertEqual(captured_payload["model"], "x-ai/grok-4.3")
        self.assertEqual(captured_payload["models"], ["x-ai/grok-4.3", "x-ai/grok-4.20"])

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

    def test_reddit_oauth_401_falls_back_to_rss(self):
        test_case = self
        rss_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/">
  <entry>
    <id>t3_redditfallback</id>
    <title>Fallback post</title>
    <category term="ClaudeAI" label="r/ClaudeAI" />
    <content type="html">&lt;p&gt;Fallback content&lt;/p&gt;</content>
    <link href="https://www.reddit.com/r/ClaudeAI/comments/redditfallback/fallback_post/" />
  </entry>
</feed>"""

        class FakeResponse:
            def __init__(self, status_code=200, text="", json_data=None):
                self.status_code = status_code
                self.text = text
                self._json_data = json_data or {}
                self.request = run.httpx.Request("GET", "https://example.test")

            def raise_for_status(self):
                if self.status_code >= 400:
                    response = run.httpx.Response(self.status_code, request=self.request, text=self.text)
                    raise run.httpx.HTTPStatusError("fake reddit error", request=self.request, response=response)

            def json(self):
                return self._json_data

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, **kwargs):
                test_case.assertEqual(url, run.REDDIT_TOKEN_URL)
                return FakeResponse(status_code=401, text="Unauthorized")

            def get(self, url, **kwargs):
                if url == run.REDDIT_PUBLIC_MULTIREDDIT:
                    return FakeResponse(status_code=403, text="Blocked")
                test_case.assertEqual(url, run.REDDIT_RSS_MULTIREDDIT)
                return FakeResponse(text=rss_feed)

        with patch.dict(os.environ, {"REDDIT_CLIENT_ID": "bad-id", "REDDIT_CLIENT_SECRET": "bad-secret"}, clear=False), \
            patch.object(run.httpx, "Client", FakeClient):
            posts = run.fetch_reddit_posts()

        self.assertEqual(posts[0]["id"], "redditfallback")
        self.assertEqual(posts[0]["subreddit"], "ClaudeAI")

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
