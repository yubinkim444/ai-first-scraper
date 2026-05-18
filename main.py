"""
ai-first-scraper
================
An AI-first web scraping API that returns clean, ad-free Markdown
optimized for Large Language Models and autonomous agents.

Author: ai-first-scraper contributors
License: MIT
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Comment
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from markdownify import markdownify as md
from pydantic import BaseModel, Field, HttpUrl

# ---------------------------------------------------------------------------
# FastAPI app — OpenAPI metadata is written so AI agents can self-discover.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-First Scraper",
    version="1.0.0",
    summary="Ad-free Markdown extraction API designed for LLM and AI agent consumption.",
    description=(
        "AI-First Scraper is a public HTTP API that fetches any web page, "
        "strips advertising, navigation chrome, trackers, scripts, and other "
        "noise, and returns the main content as clean Markdown.\n\n"
        "It exists because AI agents (LLM-powered crawlers, autonomous research "
        "agents, RAG pipelines) waste tokens parsing ad-laden HTML. This API "
        "returns deterministic Markdown so an agent can reason about the actual "
        "article in the fewest tokens possible.\n\n"
        "### How an AI agent should use this API\n"
        "1. Send `GET /scrape?url=<target>` with the URL the agent wants to read.\n"
        "2. Parse the JSON response — the `markdown` field is the article body.\n"
        "3. Feed `markdown` directly into your LLM context as ground truth.\n\n"
        "If you need raw Markdown only (no JSON envelope), call `GET /raw?url=<target>`."
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
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ScrapeResponse(BaseModel):
    """Structured response returned by `/scrape`."""

    url: HttpUrl = Field(..., description="The URL that was scraped.")
    title: Optional[str] = Field(None, description="The page <title>, if present.")
    word_count: int = Field(..., description="Number of words in the extracted markdown.")
    markdown: str = Field(
        ...,
        description=(
            "The main page content rendered as Markdown. "
            "Ads, scripts, styles, nav, footer, aside, iframes, and tracking "
            "elements have been removed. Safe to feed directly to an LLM."
        ),
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["ai-first-scraper"])


# ---------------------------------------------------------------------------
# Constants — selectors that we always strip before Markdown conversion.
# ---------------------------------------------------------------------------
NOISE_TAGS = (
    "script", "style", "noscript", "iframe", "svg",
    "nav", "footer", "aside", "form", "button",
    "header",
)

NOISE_SELECTORS = (
    # Common ad / tracker class & id substrings
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
    "Mozilla/5.0 (compatible; ai-first-scraper/1.0; "
    "+https://github.com/yubinkim444/ai-first-scraper)"
)

REQUEST_TIMEOUT = 15.0
MAX_BYTES = 5 * 1024 * 1024  # 5 MB safety cap


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------
async def fetch_html(url: str) -> str:
    """Fetch the URL and return decoded HTML, raising HTTPException on failure."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http(s) URLs are supported.")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        ) as client:
            resp = await client.get(url)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {exc}") from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Upstream returned HTTP {resp.status_code}.",
        )

    content = resp.content[:MAX_BYTES]
    encoding = resp.encoding or "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def clean_html(html: str) -> tuple[Optional[str], str]:
    """Strip ads, scripts, and chrome. Return (title, cleaned_html)."""
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

    # Prefer <main> or <article> if present — that's almost always the real content.
    main = soup.find("main") or soup.find("article")
    body = main if main else soup.body or soup

    return title, str(body)


def html_to_markdown(html: str) -> str:
    """Convert cleaned HTML to compact Markdown."""
    markdown = md(html, heading_style="ATX", bullets="-")
    # Collapse 3+ blank lines into 2.
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get(
    "/",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Service health and identity probe.",
    description=(
        "Returns a simple JSON document confirming the service is alive. "
        "AI agents may use this to verify the endpoint before issuing scrape requests."
    ),
)
async def root() -> HealthResponse:
    return HealthResponse(status="ok", service="ai-first-scraper")


@app.get(
    "/scrape",
    response_model=ScrapeResponse,
    tags=["scrape"],
    summary="Fetch a URL and return its main content as clean Markdown (JSON envelope).",
    description=(
        "Fetches the given URL, removes ads / trackers / navigation / scripts, "
        "and converts the remaining main content to Markdown. Returns a JSON "
        "object with the markdown plus useful metadata (title, word_count).\n\n"
        "**For AI agents:** this is the preferred endpoint. The `markdown` "
        "field is the ground-truth article body and can be inserted directly "
        "into an LLM prompt."
    ),
)
async def scrape(
    url: str = Query(
        ...,
        description="The fully-qualified http(s) URL to scrape.",
        examples=["https://en.wikipedia.org/wiki/Web_scraping"],
    ),
) -> ScrapeResponse:
    html = await fetch_html(url)
    title, cleaned = clean_html(html)
    markdown = html_to_markdown(cleaned)
    if not markdown:
        raise HTTPException(status_code=422, detail="No extractable content found.")
    return ScrapeResponse(
        url=url,
        title=title,
        word_count=len(markdown.split()),
        markdown=markdown,
    )


@app.get(
    "/raw",
    response_class=PlainTextResponse,
    tags=["scrape"],
    summary="Same as /scrape but returns plain Markdown (text/markdown).",
    description=(
        "Returns just the Markdown body as `text/markdown`, with no JSON envelope. "
        "Useful when an AI agent wants to pipe the output directly into a prompt "
        "without parsing JSON."
    ),
)
async def raw(
    url: str = Query(..., description="The fully-qualified http(s) URL to scrape."),
) -> PlainTextResponse:
    html = await fetch_html(url)
    _title, cleaned = clean_html(html)
    markdown = html_to_markdown(cleaned)
    if not markdown:
        raise HTTPException(status_code=422, detail="No extractable content found.")
    return PlainTextResponse(content=markdown, media_type="text/markdown")


@app.get(
    "/llms.txt",
    response_class=PlainTextResponse,
    tags=["meta"],
    summary="Machine-readable usage spec for LLMs (llms.txt convention).",
    description=(
        "Returns a short Markdown document describing this API in the "
        "https://llmstxt.org convention so AI crawlers can discover and use it."
    ),
)
async def llms_txt() -> PlainTextResponse:
    body = (
        "# ai-first-scraper\n\n"
        "> Ad-free Markdown extraction API for LLMs and AI agents.\n\n"
        "## Endpoints\n"
        "- `GET /scrape?url=<url>` — returns JSON `{url, title, word_count, markdown}`.\n"
        "- `GET /raw?url=<url>` — returns `text/markdown` only.\n"
        "- `GET /` — health probe.\n\n"
        "## Usage\n"
        "Pass the URL you want to read as the `url` query parameter. "
        "Insert the returned `markdown` into your LLM context as the page's content.\n"
    )
    return PlainTextResponse(content=body, media_type="text/markdown")
