"""
AI News Monitor - Vercel Serverless Function
Webhook endpoint triggered by GitHub Actions cron job.
Fetches Reddit AI/LLM news, generates summary via OpenRouter, publishes to Notion.
"""

import json
import os
import hmac
import hashlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler

import httpx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDDIT_MULTIREDDIT = "https://oauth.reddit.com/user/bowtiedswan/m/aillms.json"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
USER_AGENT = "ai-news-monitor/2.0 (by /u/ai-news-bot)"
MAX_POSTS = 20

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
DEFAULT_FALLBACK_MODELS = "google/gemma-4-31b-it:free,openrouter/free"
OPENROUTER_FALLBACK_MODELS = [
    model.strip()
    for model in os.environ.get("OPENROUTER_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS).split(",")
    if model.strip()
]
OPENROUTER_TIMEOUT_SECONDS = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "40"))

DEFAULT_NOTION_DATABASE_ID = "21d90be5f44d80ffa169cbb40567085b"

RELEVANT_TAGS = [
    "claude", "cursor", "mcp", "agents", "tutorial", "review",
    "release", "launch", "openai", "gpt", "llm", "anthropic",
    "gemini", "mistral", "llama", "copilot", "ai-coding", "rag",
    "fine-tuning", "prompt-engineering", "embeddings", "vector",
]


def get_text_value(data: dict[str, object], key: str, default: str = "") -> str:
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def get_int_value(data: dict[str, object], key: str, default: int = 0) -> int:
    value = data.get(key, default)
    return value if isinstance(value, int) else default


def raise_notion_error(resp: httpx.Response, action: str) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise RuntimeError(
            f"Notion {action} failed with HTTP {exc.response.status_code}: {detail}"
        ) from exc


# ---------------------------------------------------------------------------
# Reddit helpers
# ---------------------------------------------------------------------------
def _get_reddit_token() -> str:
    """Obtain an OAuth2 app-only token from Reddit."""
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set")

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            REDDIT_TOKEN_URL,
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


def fetch_reddit_posts() -> list[dict[str, object]]:
    """Fetch posts from the Reddit multireddit via OAuth."""
    token = _get_reddit_token()
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            REDDIT_MULTIREDDIT,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()

    return [child["data"] for child in data.get("data", {}).get("children", [])]


def filter_relevant_posts(posts: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove image/video/meme posts, keep text and link posts."""
    image_domains = {
        "i.redd.it", "i.imgur.com", "imgur.com", "gfycat.com",
        "v.redd.it", "reddit.com/gallery", "giphy.com", "tenor.com",
    }
    meme_flair = ["meme", "joke", "funny", "humor", "shitpost"]
    filtered = []

    for p in posts:
        if p.get("is_video"):
            continue
        domain = get_text_value(p, "domain")
        if any(d in domain for d in image_domains):
            continue
        if get_text_value(p, "post_hint") in ("image", "hosted:video", "rich:video"):
            continue
        flair = get_text_value(p, "link_flair_text").lower()
        if any(m in flair for m in meme_flair):
            continue
        if p.get("is_gallery"):
            continue
        filtered.append(p)

    return filtered


def rank_posts(posts: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort by engagement: upvotes + 2*comments."""
    return sorted(
        posts,
        key=lambda p: get_int_value(p, "ups") + get_int_value(p, "num_comments") * 2,
        reverse=True,
    )


def extract_tags(post: dict[str, object]) -> list[str]:
    """Find relevant tags in title + selftext."""
    text = f"{get_text_value(post, 'title')} {get_text_value(post, 'selftext')}".lower()
    return list({t for t in RELEVANT_TAGS if t in text})


def prepare_posts_text(posts: list[dict[str, object]], max_posts: int = MAX_POSTS) -> str:
    """Format top posts as markdown for the LLM prompt."""
    lines: list[str] = []
    for i, p in enumerate(posts[:max_posts], 1):
        title = get_text_value(p, "title", "No title")
        url = f"https://reddit.com{get_text_value(p, 'permalink')}"
        selftext = get_text_value(p, "selftext")[:500]
        ups = get_int_value(p, "ups")
        comments = get_int_value(p, "num_comments")
        ext_url = get_text_value(p, "url")
        subreddit = get_text_value(p, "subreddit")
        tags = extract_tags(p)

        lines.append(f"""
### Post {i}: {title}
- **Subreddit**: r/{subreddit}
- **Upvotes**: {ups} | **Comments**: {comments}
- **Reddit Link**: {url}
- **External Link**: {ext_url if ext_url != url else 'N/A'}
- **Detected Tags**: {', '.join(tags) if tags else 'None'}
- **Preview**: {selftext[:200] + '...' if len(selftext) > 200 else selftext or 'No text content'}
""")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenRouter LLM
# ---------------------------------------------------------------------------
def generate_summary(posts_data: str, date_str: str) -> str:
    """Call OpenRouter to generate the news summary."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    prompt = f"""You are an AI news analyst. Analyze the following Reddit posts from AI/LLM communities and create a comprehensive daily summary.

## Today's Date: {date_str}

## Posts to Analyze:
{posts_data}

## Your Task:
Create a well-structured markdown summary with the following sections:

1. **Executive Summary** (2-3 sentences highlighting the most important developments)

2. **Top Stories** (Pick the 5-7 most significant posts)
   - For each: Brief description, why it matters, and the link

3. **Trends & Themes** (What patterns do you see across posts?)

4. **Actionable Insights** (What should readers do based on this news?)
   - Tools to try
   - Techniques to learn
   - Things to watch out for

5. **Quick Links** (Categorized list of all relevant links)

6. **Tags** (Comma-separated list of all relevant tags from: claude, cursor, mcp, agents, tutorial, review, release, launch, openai, gpt, llm, anthropic, gemini, mistral, llama, copilot, ai-coding, rag, fine-tuning, prompt-engineering, embeddings, vector)

Format your response as clean markdown that can be saved directly to a file.
Start with a title: # AI Alpha - {date_str}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MorpheusAIs/ai-news-monitor",
        "X-Title": "AI News Monitor",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI news analyst specializing in AI/ML developments. "
                "Focus on accuracy, technical depth, and actionable insights. "
                "Format all output as clean markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    models = list(dict.fromkeys([OPENROUTER_MODEL, *OPENROUTER_FALLBACK_MODELS]))[:3]
    payload = {
        "model": OPENROUTER_MODEL,
        "models": models,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4000,
    }

    with httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            retry_after = exc.response.headers.get("Retry-After")
            detail = exc.response.text[:200]
            if exc.response.status_code == 429 and retry_after:
                detail = f"rate limited; retry after {retry_after}s; {detail}"
            raise RuntimeError(
                f"OpenRouter summary failed with HTTP {exc.response.status_code} using fallback chain {models}: {detail}"
            ) from exc

        data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(data)[:200]}")

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned empty content")

    return content


# ---------------------------------------------------------------------------
# Notion publishing
# ---------------------------------------------------------------------------
def markdown_to_notion_blocks(md: str) -> list[dict[str, object]]:
    """Convert markdown to Notion blocks (simplified)."""
    blocks: list[dict[str, object]] = []
    for line in md.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# "):
            blocks.append({
                "object": "block", "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": stripped[2:]}}]},
            })
        elif stripped.startswith("## "):
            blocks.append({
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": stripped[3:]}}]},
            })
        elif stripped.startswith("### "):
            blocks.append({
                "object": "block", "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": stripped[4:]}}]},
            })
        elif stripped.startswith(("- ", "* ")):
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": stripped[2:]}}]},
            })
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)":
            blocks.append({
                "object": "block", "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [{"type": "text", "text": {"content": stripped[2:].strip()}}]},
            })
        elif stripped in ("---", "***", "___"):
            blocks.append({"object": "block", "type": "divider", "divider": {}})
        else:
            text = stripped[:2000]
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            })

    return blocks[:100]  # Notion limit


def ai_alpha_page_exists(api_key: str, db_id: str, date_str: str) -> bool:
    """Check if the AI Alpha page for this date already exists."""
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
                        {"property": "Title", "title": {"contains": "AI Alpha"}},
                    ]
                },
                "page_size": 1,
            },
        )
        raise_notion_error(resp, "database query")
        return len(resp.json().get("results", [])) > 0


def publish_to_notion(content: str, date_str: str) -> str:
    """Publish summary to Notion database. Returns page URL. Skips if date exists."""
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY not set")

    db_id = os.environ.get("NOTION_DATABASE_ID", DEFAULT_NOTION_DATABASE_ID)
    title = f"AI Alpha - {date_str}"

    if ai_alpha_page_exists(api_key, db_id, date_str):
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


# ---------------------------------------------------------------------------
# Webhook authentication
# ---------------------------------------------------------------------------
def verify_webhook(secret: str, body: bytes, signature: str) -> bool:
    """Verify HMAC-SHA256 webhook signature."""
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> dict[str, object]:
    """Execute the full news monitoring pipeline. Returns status dict."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log: list[str] = [f"Starting AI News Monitor run for {date_str}"]
    skip_notion = os.environ.get("SKIP_NOTION", "").lower() == "true"
    notion_key = os.environ.get("NOTION_API_KEY")
    db_id = os.environ.get("NOTION_DATABASE_ID", DEFAULT_NOTION_DATABASE_ID)

    if not skip_notion and notion_key and ai_alpha_page_exists(notion_key, db_id, date_str):
        title = f"AI Alpha - {date_str}"
        log.append(f"Skipped before generation: page '{title}' already exists")
        return {"status": "ok", "date": date_str, "notion_url": f"SKIPPED: page '{title}' already exists", "log": log}

    # 1. Fetch
    all_posts = fetch_reddit_posts()
    log.append(f"Fetched {len(all_posts)} posts from Reddit")

    if not all_posts:
        return {"status": "ok", "message": "No posts found", "log": log}

    # 2. Filter
    relevant = filter_relevant_posts(all_posts)
    log.append(f"{len(relevant)} posts after filtering (removed {len(all_posts) - len(relevant)})")

    if not relevant:
        return {"status": "ok", "message": "No relevant posts", "log": log}

    # 3. Rank
    ranked = rank_posts(relevant)
    if ranked:
        log.append(f"Top post: {get_text_value(ranked[0], 'title', '?')[:60]}")

    # 4. Generate summary
    posts_data = prepare_posts_text(ranked)
    summary = generate_summary(posts_data, date_str)
    log.append(f"Generated summary: {len(summary)} chars")

    # 5. Publish to Notion
    notion_url = ""

    if skip_notion:
        log.append("Skipped Notion publishing (SKIP_NOTION=true)")
    elif not notion_key:
        log.append("Skipped Notion publishing (NOTION_API_KEY not set)")
    else:
        try:
            notion_url = publish_to_notion(summary, date_str)
            log.append(f"Published to Notion: {notion_url}")
        except Exception as exc:
            log.append(f"Notion publishing failed: {exc}")
            raise RuntimeError(f"Notion publishing failed: {exc}") from exc

    return {
        "status": "ok",
        "date": date_str,
        "posts_fetched": len(all_posts),
        "posts_relevant": len(relevant),
        "summary_length": len(summary),
        "summary": summary,
        "notion_url": notion_url,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Vercel handler
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Health check."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "service": "ai-news-monitor"}).encode())

    def do_POST(self):
        """Webhook endpoint — triggers the news pipeline."""
        # Read body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Authenticate
        webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
        if webhook_secret:
            sig = self.headers.get("X-Webhook-Signature", "")
            if not verify_webhook(webhook_secret, body, sig):
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid signature"}).encode())
                return

        # Run pipeline
        try:
            result = run_pipeline()
            status_code = 200
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
            status_code = 500

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode())
