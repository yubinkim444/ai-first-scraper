"""
ai-first-scraper
================
An AI-first web scraping API that returns clean, ad-free Markdown
optimized for Large Language Models and autonomous agents.

Endpoints:
    GET  /scrape        single URL  -> JSON {url, title, word_count, markdown, links}
    GET  /raw           single URL  -> text/markdown
    POST /batch         many URLs   -> JSON [{url, ok, ...}, ...]
    GET  /              health probe
    GET  /llms.txt      self-describing spec for LLM crawlers
    GET  /openapi.json  full OpenAPI spec

License: MIT
"""

from __future__ import annotations

import asyncio
import io
import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Comment
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from markdownify import markdownify as md
from pydantic import BaseModel, Field, HttpUrl
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# FastAPI app — OpenAPI metadata is written so AI agents can self-discover.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-First Scraper",
    version="1.1.0",
    summary="Ad-free Markdown extraction API designed for LLM and AI agent consumption.",
    description=(
        "AI-First Scraper fetches any web page or PDF, strips advertising, "
        "navigation chrome, trackers, and scripts, and returns the main "
        "content as clean Markdown.\n\n"
        "It exists because AI agents (LLM-powered crawlers, autonomous research "
        "agents, RAG pipelines) waste tokens parsing ad-laden HTML. This API "
        "returns deterministic Markdown so an agent can reason about the actual "
        "article in the fewest tokens possible.\n\n"
        "### How an AI agent should use this API\n"
        "1. **Single URL** — `GET /scrape?url=<target>` (JSON) or `/raw?url=<target>` (markdown).\n"
        "2. **Many URLs** — `POST /batch` with `{\"urls\": [...], \"max_tokens\": N}` "
        "for parallel fetch.\n"
        "3. Feed `markdown` directly into your LLM context.\n\n"
        "All endpoints support `max_tokens` to cap the response size and protect "
        "your prompt budget."
    ),
    contact={
        "name": "ai-first-scraper",
        "url": "https://github.com/yubinkim444/ai-first-scraper",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ScrapeResponse(BaseModel):
    """Structured response returned by `/scrape` and per-item in `/batch`."""

    url: HttpUrl = Field(..., description="The URL that was scraped.")
    title: Optional[str] = Field(None, description="The page <title>, if present.")
    word_count: int = Field(..., description="Number of words in the extracted markdown.")
    markdown: str = Field(
        ...,
        description=(
            "The main page content rendered as Markdown. Ads, scripts, styles, "
            "nav, footer, aside, iframes, and tracking elements have been removed."
        ),
    )
    links: list[str] = Field(
        default_factory=list,
        description=(
            "All outbound HTTP(S) links found in the cleaned content, deduplicated "
            "and in document order. Useful for agents that need to plan the next hop."
        ),
    )
    truncated: bool = Field(
        False,
        description="True when the `markdown` was cut off because it exceeded `max_tokens`.",
    )
    content_type: str = Field(
        "html",
        description="Either 'html' or 'pdf' depending on what the upstream returned.",
    )


class BatchItem(BaseModel):
    url: str
    ok: bool
    data: Optional[ScrapeResponse] = None
    error: Optional[str] = None


class BatchRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=25,
        description="Up to 25 URLs to scrape in parallel.",
        examples=[["https://example.com", "https://en.wikipedia.org/wiki/AI"]],
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=100,
        description=(
            "Per-URL soft cap on the markdown size, measured in whitespace-split "
            "tokens. Pages larger than this are truncated and `truncated=true` is set."
        ),
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["ai-first-scraper"])
    version: str = Field(..., examples=["1.1.0"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOISE_TAGS = (
    "script", "style", "noscript", "iframe", "svg",
    "nav", "footer", "aside", "form", "button",
    "header",
)

NOISE_SELECTORS = (
    '[class*="ad-"]', '[class*="-ad"]', '[class*="ads"]',
    '[id*="ad-"]', '[id*="-ad"]', '[id*="ads"]',
    '[class*="advert"]', '[id*="advert"]',
    '[class*="banner"]', '[id*="banner"]',
    '[class*="sponsor"]', '[id*="sponsor"]',
    '[class*="promo"]', '[id*="promo"]',
    '[class*="popup"]', '[id*="popup"]',
    '[class*="cookie"]', '[id*="cookie"]',
    '[class*="newsletter"]', '[id*="newsletter"]',
    '[class*="social"]', '[id*="social"]',
    '[class*="share"]', '[id*="share"]',
    '[class*="related"]', '[id*="related"]',
    '[class*="recommend"]', '[id*="recommend"]',
    '[class*="comment"]', '[id*="comment"]',
    '[role="banner"]', '[role="navigation"]', '[role="complementary"]',
    "[aria-hidden='true']",
)

USER_AGENT = (
    "Mozilla/5.0 (compatible; ai-first-scraper/1.1; "
    "+https://github.com/yubinkim444/ai-first-scraper)"
)

REQUEST_TIMEOUT = 15.0
MAX_BYTES = 10 * 1024 * 1024  # 10 MB cap (PDFs can be large)
LINK_RE = re.compile(r"^https?://", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
async def fetch(url: str) -> tuple[bytes, str]:
    """Fetch URL → (content_bytes, content_type). Raises HTTPException on failure."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http(s) URLs are supported.")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,*/*"},
        ) as client:
            resp = await client.get(url)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {exc!s}") from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Upstream returned HTTP {resp.status_code}.",
        )

    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    content = resp.content[:MAX_BYTES]
    return content, ctype


# ---------------------------------------------------------------------------
# HTML → Markdown
# ---------------------------------------------------------------------------
def clean_html(html: str) -> tuple[Optional[str], str, list[str]]:
    """Strip ads, scripts, chrome. Return (title, cleaned_html, links)."""
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for selector in NOISE_SELECTORS:
        try:
            for tag in soup.select(selector):
                tag.decompose()
        except Exception:
            continue

    main = soup.find("main") or soup.find("article")
    body = main if main else soup.body or soup

    seen: set[str] = set()
    links: list[str] = []
    for a in body.find_all("a", href=True):
        href = a["href"].strip()
        if LINK_RE.match(href) and href not in seen:
            seen.add(href)
            links.append(href)

    return title, str(body), links


def html_to_markdown(html: str) -> str:
    markdown = md(html, heading_style="ATX", bullets="-")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


# ---------------------------------------------------------------------------
# PDF → Markdown
# ---------------------------------------------------------------------------
def pdf_to_markdown(content: bytes) -> tuple[Optional[str], str]:
    """Return (title, markdown) from PDF bytes."""
    reader = PdfReader(io.BytesIO(content))
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or None

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text.strip())

    body = "\n\n".join(pages)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return title, body


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------
def truncate(markdown: str, max_tokens: Optional[int]) -> tuple[str, bool]:
    if max_tokens is None:
        return markdown, False
    tokens = markdown.split()
    if len(tokens) <= max_tokens:
        return markdown, False
    return " ".join(tokens[:max_tokens]) + "\n\n[...truncated]", True


# ---------------------------------------------------------------------------
# Shared scrape pipeline
# ---------------------------------------------------------------------------
async def scrape_one(url: str, max_tokens: Optional[int]) -> ScrapeResponse:
    content, ctype = await fetch(url)
    links: list[str] = []
    if "pdf" in ctype:
        title, body = pdf_to_markdown(content)
        kind = "pdf"
    else:
        encoding = "utf-8"
        try:
            text = content.decode(encoding, errors="replace")
        except LookupError:
            text = content.decode("utf-8", errors="replace")
        title, cleaned, links = clean_html(text)
        body = html_to_markdown(cleaned)
        kind = "html"

    if not body:
        raise HTTPException(status_code=422, detail="No extractable content found.")

    body, was_truncated = truncate(body, max_tokens)
    return ScrapeResponse(
        url=url,
        title=title,
        word_count=len(body.split()),
        markdown=body,
        links=links,
        truncated=was_truncated,
        content_type=kind,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, tags=["meta"], summary="Liveness probe.")
async def root() -> HealthResponse:
    return HealthResponse(status="ok", service="ai-first-scraper", version="1.1.0")


@app.get(
    "/scrape",
    response_model=ScrapeResponse,
    tags=["scrape"],
    summary="Fetch one URL and return clean Markdown (JSON).",
    description=(
        "Fetches a single URL (HTML or PDF), removes ads / trackers / nav / scripts, "
        "and returns Markdown plus metadata (title, word_count, links). "
        "Use `max_tokens` to cap the body size."
    ),
)
async def scrape(
    url: str = Query(..., description="Fully-qualified http(s) URL.", examples=["https://en.wikipedia.org/wiki/Web_scraping"]),
    max_tokens: Optional[int] = Query(None, ge=100, description="Soft cap on the returned markdown (whitespace tokens)."),
) -> ScrapeResponse:
    return await scrape_one(url, max_tokens)


@app.get(
    "/raw",
    response_class=PlainTextResponse,
    tags=["scrape"],
    summary="Same as /scrape but returns plain text/markdown (no JSON envelope).",
)
async def raw(
    url: str = Query(...),
    max_tokens: Optional[int] = Query(None, ge=100),
) -> PlainTextResponse:
    data = await scrape_one(url, max_tokens)
    return PlainTextResponse(content=data.markdown, media_type="text/markdown")


@app.post(
    "/batch",
    response_model=list[BatchItem],
    tags=["scrape"],
    summary="Fetch many URLs in parallel and return per-URL results.",
    description=(
        "Accepts up to 25 URLs and scrapes them concurrently. Returns an array in "
        "the same order as the input — each item has `ok` plus either `data` "
        "(success) or `error` (failure). One failing URL never blocks the others."
    ),
)
async def batch(req: BatchRequest) -> list[BatchItem]:
    async def one(u: str) -> BatchItem:
        try:
            data = await scrape_one(u, req.max_tokens)
            return BatchItem(url=u, ok=True, data=data)
        except HTTPException as exc:
            return BatchItem(url=u, ok=False, error=f"HTTP {exc.status_code}: {exc.detail}")
        except Exception as exc:
            return BatchItem(url=u, ok=False, error=f"{type(exc).__name__}: {exc}")

    return await asyncio.gather(*(one(u) for u in req.urls))


@app.get(
    "/robots.txt",
    response_class=PlainTextResponse,
    tags=["meta"],
    summary="Robots policy — explicitly welcomes AI / LLM crawlers.",
    include_in_schema=False,
)
async def robots_txt() -> PlainTextResponse:
    body = (
        "User-agent: GPTBot\nAllow: /\n\n"
        "User-agent: ChatGPT-User\nAllow: /\n\n"
        "User-agent: ClaudeBot\nAllow: /\n\n"
        "User-agent: anthropic-ai\nAllow: /\n\n"
        "User-agent: Claude-Web\nAllow: /\n\n"
        "User-agent: PerplexityBot\nAllow: /\n\n"
        "User-agent: Google-Extended\nAllow: /\n\n"
        "User-agent: CCBot\nAllow: /\n\n"
        "User-agent: cohere-ai\nAllow: /\n\n"
        "User-agent: *\nAllow: /\n\n"
        "Sitemap: https://ai-first-scraper.onrender.com/sitemap.xml\n"
    )
    return PlainTextResponse(content=body, media_type="text/plain")


@app.get(
    "/sitemap.xml",
    tags=["meta"],
    summary="Sitemap for AI search crawlers (Perplexity, ChatGPT browse, etc.).",
    include_in_schema=False,
)
async def sitemap() -> Response:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        '  <url><loc>https://ai-first-scraper.onrender.com/</loc></url>\n'
        '  <url><loc>https://ai-first-scraper.onrender.com/docs</loc></url>\n'
        '  <url><loc>https://ai-first-scraper.onrender.com/openapi.json</loc></url>\n'
        '  <url><loc>https://ai-first-scraper.onrender.com/llms.txt</loc></url>\n'
        '  <url><loc>https://ai-first-scraper.onrender.com/.well-known/ai-plugin.json</loc></url>\n'
        '</urlset>\n'
    )
    return Response(content=body, media_type="application/xml")


@app.get(
    "/.well-known/ai-plugin.json",
    tags=["meta"],
    summary="AI plugin manifest (ChatGPT / agent crawler discovery).",
    include_in_schema=False,
)
async def ai_plugin() -> dict:
    return {
        "schema_version": "v1",
        "name_for_human": "AI-First Scraper",
        "name_for_model": "ai_first_scraper",
        "description_for_human": "Ad-free Markdown extraction API designed for LLM agents.",
        "description_for_model": (
            "Use this tool whenever you need to read the content of any URL. "
            "It fetches the page (HTML or PDF), strips advertising, navigation, "
            "scripts, and trackers, and returns clean Markdown that you can "
            "feed directly into your reasoning. Supports per-page max_tokens "
            "and batch fetching of up to 25 URLs in one request."
        ),
        "auth": {"type": "none"},
        "api": {
            "type": "openapi",
            "url": "https://ai-first-scraper.onrender.com/openapi.json",
            "has_user_authentication": False,
        },
        "logo_url": "https://ai-first-scraper.onrender.com/",
        "contact_email": "noreply@example.com",
        "legal_info_url": "https://github.com/yubinkim444/ai-first-scraper/blob/main/LICENSE",
    }


@app.get(
    "/llms.txt",
    response_class=PlainTextResponse,
    tags=["meta"],
    summary="Machine-readable usage spec for LLMs (llms.txt convention).",
)
async def llms_txt() -> PlainTextResponse:
    body = (
        "# ai-first-scraper\n\n"
        "> Ad-free Markdown extraction API for LLMs and AI agents. HTML + PDF.\n\n"
        "## Endpoints\n"
        "- `GET  /scrape?url=<url>[&max_tokens=N]` — JSON `{url, title, word_count, markdown, links, truncated, content_type}`.\n"
        "- `GET  /raw?url=<url>[&max_tokens=N]` — `text/markdown` only.\n"
        "- `POST /batch` body `{\"urls\":[...], \"max_tokens\": N?}` — array of per-URL results.\n"
        "- `GET  /openapi.json` — full machine-readable spec.\n"
        "- `GET  /` — liveness.\n\n"
        "## Limits\n"
        "- Up to 25 URLs per batch.\n"
        "- 10 MB upstream cap, 15s timeout per URL.\n"
        "- `max_tokens` is a soft cap on whitespace-split tokens.\n"
    )
    return PlainTextResponse(content=body, media_type="text/markdown")
