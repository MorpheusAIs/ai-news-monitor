"""
Grok Alpha - Vercel Serverless Function
Standalone endpoint for Grok-powered AI & Tech news research.
Uses x-ai/grok-4.1-fast via OpenRouter with reasoning enabled.
No Reddit dependency — Grok performs its own research.
"""

import json
import os
import hmac
import hashlib
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler
from typing import Any

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_GROK_MODEL = "x-ai/grok-4.3"
DEFAULT_GROK_FALLBACK_MODELS = "x-ai/grok-4.20"
GROK_TIMEOUT_SECONDS = float(os.environ.get("GROK_TIMEOUT_SECONDS", "55"))
DEFAULT_NOTION_DATABASE_ID = "21d90be5f44d80ffa169cbb40567085b"


def raise_notion_error(resp: httpx.Response, action: str) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise RuntimeError(
            f"Notion {action} failed with HTTP {exc.response.status_code}: {detail}"
        ) from exc


def generate_grok_report(date_str: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    primary_model = os.environ.get("GROK_MODEL", DEFAULT_GROK_MODEL).strip() or DEFAULT_GROK_MODEL
    fallback_models = [
        model.strip()
        for model in os.environ.get("GROK_FALLBACK_MODELS", DEFAULT_GROK_FALLBACK_MODELS).split(",")
        if model.strip()
    ]
    models = list(dict.fromkeys([primary_model, *fallback_models]))[:3]

    end_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days=1)

    prompt = (
        "Summarize the most important AI & Tech developments from the past 24 hours, "
        "including new tools, updates, and announcements. Prioritize model releases, "
        "new papers, viral X posts or threads (that either show an authors project or "
        "a new open source project, announcement, a breakthrough, a cool implementation etc) "
        "and open-source projects and include source links from web searches, X posts, etc. "
        "For every key point or example sourced from X, include the actual https://x.com/... post link, "
        "author handle, and post date. Use only real data returned by search tools. "
        "Organize the information to be easily digestible and readable.\n\n"
        f"Today's date: {date_str}\n\n"
        "Format your response as clean markdown.\n"
        f"Start with a title: # Grok Alpha - {date_str}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MorpheusAIs/ai-news-monitor",
        "X-Title": "AI News Monitor - Grok Alpha",
    }

    payload = {
        "model": primary_model,
        "models": models,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert AI & Tech news analyst with access to the latest information. "
                    "Provide comprehensive, accurate analysis of the most important AI/ML developments, "
                    "new releases, viral posts, and emerging trends from the past 24 hours. "
                    "Include real source links (X/Twitter posts, GitHub repos, blog posts, arxiv papers). "
                    "When citing X content, include the actual https://x.com/... post URL, author handle, and date, "
                    "and do not invent or paraphrase links. "
                    "Format all output as clean, well-structured markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "plugins": [{"id": "web"}],
        "x_search_filter": {
            "from_date": start_date.strftime("%Y-%m-%d"),
            "to_date": end_date.strftime("%Y-%m-%d"),
        },
        "reasoning": {
            "enabled": True,
        },
        "temperature": 0.7,
        "max_tokens": 8000,
    }

    with httpx.Client(timeout=GROK_TIMEOUT_SECONDS) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            raise RuntimeError(
                f"OpenRouter Grok report failed with HTTP {exc.response.status_code} using fallback chain {models}: {detail}"
            ) from exc
        data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(data)}")

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned empty content")

    return content


def markdown_to_notion_blocks(md: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for line in md.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# "):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [
                            {"type": "text", "text": {"content": stripped[2:]}}
                        ]
                    },
                }
            )
        elif stripped.startswith("## "):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [
                            {"type": "text", "text": {"content": stripped[3:]}}
                        ]
                    },
                }
            )
        elif stripped.startswith("### "):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [
                            {"type": "text", "text": {"content": stripped[4:]}}
                        ]
                    },
                }
            )
        elif stripped.startswith(("- ", "* ")):
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [
                            {"type": "text", "text": {"content": stripped[2:]}}
                        ]
                    },
                }
            )
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)":
            blocks.append(
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [
                            {"type": "text", "text": {"content": stripped[2:].strip()}}
                        ]
                    },
                }
            )
        elif stripped in ("---", "***", "___"):
            blocks.append({"object": "block", "type": "divider", "divider": {}})
        else:
            text = stripped[:2000]
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": text}}]
                    },
                }
            )

    return blocks[:100]


def grok_page_exists(api_key: str, db_id: str, date_str: str) -> bool:
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            f"https://api.notion.com/v1/databases/{db_id}/query",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            json={
                "filter": {
                    "and": [
                        {"property": "datetime", "date": {"equals": date_str}},
                        {"property": "Title", "title": {"contains": "Grok Alpha"}},
                    ]
                },
                "page_size": 1,
            },
        )
        raise_notion_error(resp, "database query")
        return len(resp.json().get("results", [])) > 0


def publish_to_notion(content: str, date_str: str) -> str:
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY not set")

    db_id = os.environ.get("NOTION_DATABASE_ID", DEFAULT_NOTION_DATABASE_ID)
    title = f"Grok Alpha - {date_str}"
    if grok_page_exists(api_key, db_id, date_str):
        return f"SKIPPED: page '{title}' already exists"

    blocks = markdown_to_notion_blocks(content)

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            json={
                "parent": {"database_id": db_id},
                "properties": {
                    "Title": {"title": [{"text": {"content": title}}]},
                    "datetime": {"date": {"start": date_str}},
                },
                "children": blocks,
            },
        )
        raise_notion_error(resp, "page create")
        page = resp.json()

    url = page.get("url", "")
    return url if isinstance(url, str) else ""


def verify_webhook(secret: str, body: bytes, signature: str) -> bool:
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def run_grok_pipeline() -> dict[str, Any]:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log: list[str] = [f"Starting Grok Alpha run for {date_str}"]

    report = generate_grok_report(date_str)
    log.append(f"Generated Grok report (reasoning enabled): {len(report)} chars")

    notion_url = ""
    skip_notion = os.environ.get("SKIP_NOTION", "").lower() == "true"
    notion_key = os.environ.get("NOTION_API_KEY")

    if skip_notion:
        log.append("Skipped Notion publishing (SKIP_NOTION=true)")
    elif not notion_key:
        log.append("Skipped Notion publishing (NOTION_API_KEY not set)")
    else:
        try:
            notion_url = publish_to_notion(report, date_str)
            log.append(f"Published to Notion: {notion_url}")
        except Exception as exc:
            log.append(f"Notion publishing failed: {exc}")
            raise RuntimeError(f"Notion publishing failed: {exc}") from exc

    return {
        "status": "ok",
        "job": "grok-alpha",
        "date": date_str,
        "report_length": len(report),
        "report": report,
        "notion_url": notion_url,
        "log": log,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "service": "grok-alpha"}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
        if webhook_secret:
            sig = self.headers.get("X-Webhook-Signature", "")
            if not verify_webhook(webhook_secret, body, sig):
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid signature"}).encode())
                return

        try:
            result = run_grok_pipeline()
            status_code = 200
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
            status_code = 500

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode())
