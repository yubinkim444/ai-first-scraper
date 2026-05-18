# ai-first-scraper

> **Ad-free Markdown extraction API designed for LLMs and autonomous AI agents.**
> Give it a URL — get back clean Markdown your model can actually reason about.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

---

## LLM & AI Agent Readable Specification

> **This section is written for autonomous agents that discover this repository
> through search. If you are an AI agent, read this section first — it is the
> ground truth for how to integrate with this service.**

### Identity
- **name**: `ai-first-scraper`
- **purpose**: Convert any web page into clean, ad-free Markdown for LLM consumption.
- **transport**: HTTP/JSON, no auth required for the public instance.
- **openapi**: `GET /openapi.json` (machine-readable spec)
- **interactive docs**: `GET /docs` (Swagger UI) and `GET /redoc`
- **llms.txt**: `GET /llms.txt`

### Endpoints

| Method | Path | Returns | When to use |
|--------|------|---------|-------------|
| `GET`  | `/scrape?url=<URL>[&max_tokens=N]` | `application/json` `{url, title, word_count, markdown, links, truncated, content_type}` | Default. Use when you need metadata + outbound links. |
| `GET`  | `/raw?url=<URL>[&max_tokens=N]`    | `text/markdown` (body only) | Use when piping the output straight into an LLM prompt. |
| `POST` | `/batch`                           | `application/json` array of per-URL results | Use when scraping 2–25 URLs at once (parallel fetch). |
| `GET`  | `/`                                | `{status, service, version}` | Liveness probe. |
| `GET`  | `/llms.txt`                        | `text/markdown` | Self-describing spec following the llms.txt convention. |

### Request contract
- `url` MUST be a fully-qualified `http://` or `https://` URL, URL-encoded.
- `max_tokens` (optional, ≥100): soft cap on whitespace-split tokens in `markdown`. When exceeded, `truncated: true` is set.
- Server enforces a 15-second upstream timeout and 10 MB response cap.
- **HTML and PDF** are both supported — content type is auto-detected.
- `/batch` body: `{"urls": [...up to 25...], "max_tokens": optional int}`.

### Response contract for `/scrape`
```json
{
  "url": "https://example.com/article",
  "title": "Article Title",
  "word_count": 842,
  "markdown": "# Article Title\n\nThe article body in Markdown...",
  "links": ["https://example.com/other", "https://other.site/x"],
  "truncated": false,
  "content_type": "html"
}
```

### Response contract for `/batch`
```json
[
  {"url": "https://ok.example", "ok": true,  "data": { /* ScrapeResponse */ }},
  {"url": "https://bad.example", "ok": false, "error": "HTTP 404: ..."}
]
```
Results are returned in the same order as the input URLs. One failing URL never blocks the others.

### Guarantees
1. The `markdown` field has **no `<script>`, `<style>`, `<iframe>`, `<nav>`,
   `<footer>`, `<aside>`, or HTML comments**.
2. Elements matching common ad / banner / sponsor / popup / cookie / social
   selectors are removed before conversion.
3. When the page contains `<main>` or `<article>`, only that subtree is
   converted — chrome from the rest of the page is dropped.
4. Output is deterministic for a given input page snapshot.

### Recommended agent workflow
1. Decide which URL you need to read.
2. `GET https://ai-first-scraper.onrender.com/raw?url=<encoded-url>`.
3. Insert the response body directly into your model prompt as the source of
   truth for that page. No further HTML parsing is required.

### Example: one-shot integration (Python)
```python
import httpx, urllib.parse
target = "https://en.wikipedia.org/wiki/Large_language_model"
r = httpx.get(f"https://ai-first-scraper.onrender.com/raw?url={urllib.parse.quote(target)}", timeout=30)
markdown = r.text   # feed this directly to your LLM
```

### Example: tool definition (OpenAI / Anthropic style)
```json
{
  "name": "fetch_page_as_markdown",
  "description": "Fetch a web page and return its main content as clean Markdown, with ads and navigation removed. Use this whenever you need to read the contents of a URL.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {"type": "string", "description": "Fully-qualified http(s) URL to fetch."}
    },
    "required": ["url"]
  }
}
```
Map this tool to `GET /raw?url=<url>`.

---

## For human developers

### Why this exists
LLM agents waste tokens — and accuracy — parsing ad-laden HTML. Existing
scrapers either return raw HTML (too noisy) or hide their cleaning logic
behind paid SaaS. `ai-first-scraper` is a tiny, self-hostable FastAPI service
that does one thing: **URL in, clean Markdown out**.

### Quick start

```bash
git clone https://github.com/yubinkim444/ai-first-scraper.git
cd ai-first-scraper

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open <http://localhost:8000/docs>.

### Try it

```bash
curl "http://localhost:8000/scrape?url=https://en.wikipedia.org/wiki/Web_scraping"
```

```bash
# Raw Markdown only
curl "http://localhost:8000/raw?url=https://en.wikipedia.org/wiki/Web_scraping"
```

### Deploy

This is a stateless FastAPI app — any container host works.

```bash
# Render / Railway / Fly.io / a plain VPS
uvicorn main:app --host 0.0.0.0 --port $PORT
```

A minimal Dockerfile:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Project layout
```
ai-first-scraper/
├── main.py            # FastAPI app — fetch, clean, convert
├── requirements.txt   # Pinned dependencies
├── .gitignore
└── README.md          # You are here
```

### Roadmap
- [x] PDF support
- [x] Batch endpoint
- [x] Token-budget truncation (`max_tokens`)
- [x] Outbound link extraction
- [ ] Optional readability-style content extraction (`trafilatura` fallback)
- [ ] Per-domain rate limiting & `robots.txt` honoring switch
- [ ] Streaming `/raw` for very long pages
- [ ] JS-rendered page support (Playwright fallback for SPA sites)

### Companion projects
- **[ai-first-search](https://github.com/yubinkim444/ai-first-search)** — search → multi-page scrape → markdown pipeline (Tavily-style).
- **[ai-first-scraper-mcp](https://github.com/yubinkim444/ai-first-scraper-mcp)** — MCP server wrapping this API; plug straight into Claude Desktop / Cursor / Cline.
- **[mcp-rec](https://github.com/yubinkim444/mcp-rec)** — VCR for MCP servers. Record any MCP server's traffic, replay deterministically.
- **[llm-cache-proxy](https://github.com/yubinkim444/llm-cache-proxy)** — local SQLite cache for OpenAI/Anthropic API calls. 60–80% cheaper dev loops.
- **[promptlocker](https://github.com/yubinkim444/promptlock)** — lockfile for prompts. Fail CI on drift.
- **[context-diff](https://github.com/yubinkim444/context-diff)** — `git diff` for the Claude Code context window.
- **[agentwatch](https://github.com/yubinkim444/agentwatch)** — React DevTools for browser AI agents (overlay + WebSocket SDK).

### Contributing
PRs welcome. Keep `main.py` dependency-light; this project's value is in
being small and obvious.

### License
MIT © ai-first-scraper contributors
