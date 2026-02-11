# AI News Monitor

Automated daily AI/LLM news aggregator deployed as a Vercel serverless function. Monitors Reddit communities, filters relevant content, and generates actionable summaries using OpenRouter (StepFun Step 3.5 Flash).

## How It Works

1. GitHub Actions cron job fires a webhook POST to Vercel daily at 9 AM UTC
2. Vercel serverless function fetches posts from a curated Reddit multireddit
3. Posts are filtered (no memes/images) and ranked by engagement
4. OpenRouter generates a structured markdown summary via `stepfun/step-3.5-flash:free`
5. Summary is published to a Notion database

## Setup

### 1. Deploy to Vercel

```bash
vercel --prod
```

### 2. Set Environment Variables

Via Vercel CLI or dashboard:

```bash
vercel env add OPENROUTER_API_KEY production
vercel env add WEBHOOK_SECRET production
vercel env add NOTION_API_KEY production       # optional
vercel env add NOTION_DATABASE_ID production   # optional
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | [OpenRouter](https://openrouter.ai/keys) API key |
| `WEBHOOK_SECRET` | Yes | HMAC secret shared with GitHub Actions |
| `NOTION_API_KEY` | No | [Notion integration](https://www.notion.so/my-integrations) token |
| `NOTION_DATABASE_ID` | No | Target Notion database ID (has default) |
| `SKIP_NOTION` | No | Set to `true` to disable Notion publishing |

### 3. Configure GitHub Secrets

In your repo settings, add:

- `WEBHOOK_URL` — Your Vercel function URL (e.g. `https://ai-news-monitor.vercel.app/api/run`)
- `WEBHOOK_SECRET` — Same secret used in Vercel env vars

### 4. Generate a Webhook Secret

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/run` | Health check |
| `POST` | `/api/run` | Trigger news pipeline (requires webhook signature) |

## Architecture

```
GitHub Actions (cron 9AM UTC)
  └── POST /api/run (HMAC-signed)
        ├── Fetch Reddit multireddit
        ├── Filter & rank posts
        ├── Generate summary (OpenRouter)
        └── Publish to Notion
```

## Local Development

```bash
# Install Vercel CLI
npm i -g vercel

# Link project
vercel link

# Pull env vars
vercel env pull .env

# Run locally
vercel dev
```

## License

MIT
