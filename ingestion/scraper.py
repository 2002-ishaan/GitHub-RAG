"""
GitHub Docs Knowledge Base Scraper
====================================
Scrapes publicly accessible GitHub documentation pages and saves
structured JSON documents ready for ChromaDB ingestion.

Verified sources (all confirmed public, plain HTML, no login required):
  1. Get Started     — docs.github.com/en/get-started
  2. Repositories    — docs.github.com/en/repositories
  3. Billing         — docs.github.com/en/billing
  4. Authentication  — docs.github.com/en/authentication
  5. Organizations   — docs.github.com/en/organizations
  6. Security        — docs.github.com/en/code-security

Output: data/raw/github_docs/  (one JSON per page)

Usage:
    pip install requests beautifulsoup4
    python ingestion/scraper.py
"""

import os
import json
import time
import logging
import hashlib
from urllib.parse import urljoin, urlparse
from typing import Optional
from collections import Counter

import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR      = "data/raw/github_docs"
DELAY_SECONDS   = 1.0        # polite crawl delay
MAX_PAGES       = 250        # safety cap — will get 150+ docs easily
REQUEST_TIMEOUT = 15
MIN_CONTENT_LEN = 100        # skip pages with fewer than this many characters

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicRAGBot/1.0; "
        "University of Toronto Course Project)"
    ),
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── File extensions to NEVER parse as HTML ─────────────────────────────────────
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".mp4", ".mp3", ".css", ".js", ".xml",
    ".woff", ".woff2", ".ttf", ".eot",
}

# ── Confirmed seed URLs (verified from live search results) ───────────────────
SEED_URLS = [
    # Get Started
    "https://docs.github.com/en/get-started",
    "https://docs.github.com/en/get-started/learning-about-github/types-of-github-accounts",
    "https://docs.github.com/en/get-started/learning-about-github/access-permissions-on-github",
    "https://docs.github.com/en/get-started/onboarding/getting-started-with-github-team",

    # Repositories
    "https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories",
    "https://docs.github.com/en/repositories",

    # Billing
    "https://docs.github.com/en/billing",
    "https://docs.github.com/en/billing/get-started/introduction-to-billing",
    "https://docs.github.com/en/billing/get-started/how-billing-works",

    # Authentication
    "https://docs.github.com/en/authentication",
    "https://docs.github.com/en/apps/oauth-apps/using-oauth-apps/authorizing-oauth-apps",

    # Security
    "https://docs.github.com/en/code-security/getting-started/github-security-features",
    "https://docs.github.com/en/code-security",

    # Organizations
    "https://docs.github.com/en/organizations",
]

# ── URL scope — only crawl these path prefixes on docs.github.com ──────────────
ALLOWED_PREFIXES = [
    "/en/get-started",
    "/en/repositories",
    "/en/billing",
    "/en/authentication",
    "/en/organizations",
    "/en/code-security",
    "/en/account-and-profile",
    "/en/issues",
    "/en/pull-requests",
    "/en/actions",
    "/en/pages",
]

# Never crawl these — enterprise-specific, non-English, or irrelevant
BLOCKED_FRAGMENTS = [
    "/enterprise-server@",
    "/enterprise-cloud@",
    "/en/enterprise",
    "?apiVersion",
    "/ja/", "/zh/", "/es/", "/pt/", "/fr/", "/de/", "/ru/", "/ko/",
    "github.com/login",
    "github.com/signup",
]

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_doc_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def is_binary_url(url: str) -> bool:
    """Return True if the URL points to a non-HTML file."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def is_in_scope(url: str) -> bool:
    """Only crawl docs.github.com pages within our allowed topic sections."""
    try:
        parsed = urlparse(url)

        # Must be docs.github.com
        if parsed.netloc != "docs.github.com":
            return False

        # Never parse binary files
        if is_binary_url(url):
            return False

        path = parsed.path

        # Must not contain blocked fragments
        full = url.lower()
        if any(b in full for b in BLOCKED_FRAGMENTS):
            return False

        # Must start with an allowed prefix
        return any(path.startswith(p) for p in ALLOWED_PREFIXES)

    except Exception:
        return False


def fetch(url: str) -> Optional[BeautifulSoup]:
    """GET a page, return parsed BeautifulSoup or None on any error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # Safety check — only parse text/html responses
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            log.debug("  – Skipping non-HTML content type: %s", content_type)
            return None

        return BeautifulSoup(resp.text, "html.parser")

    except Exception as e:
        log.warning("  ✗ Fetch failed: %s — %s", url, e)
        return None


def extract_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    """Return all in-scope absolute links found on this page."""
    links = []
    for tag in soup.find_all("a", href=True):
        href     = tag["href"].strip()
        absolute = urljoin(current_url, href)
        # Strip fragments and query strings for clean dedup
        clean = absolute.split("#")[0].split("?")[0].rstrip("/")
        if clean and is_in_scope(clean):
            links.append(clean)
    return list(set(links))


def detect_category(url: str) -> str:
    """Infer a readable category from the URL path."""
    path = urlparse(url).path.lower()
    mapping = {
        "get-started":       "Getting Started",
        "learning-about":    "Learning GitHub",
        "onboarding":        "Onboarding",
        "repositories":      "Repositories",
        "billing":           "Billing & Plans",
        "authentication":    "Authentication & Security",
        "oauth":             "OAuth & Apps",
        "organizations":     "Organizations & Teams",
        "code-security":     "Code Security",
        "account-and-profile": "Account & Profile",
        "issues":            "Issues",
        "pull-requests":     "Pull Requests",
        "actions":           "GitHub Actions",
        "pages":             "GitHub Pages",
    }
    for keyword, label in mapping.items():
        if keyword in path:
            return label
    return "General"


def extract_content(soup: BeautifulSoup, url: str) -> Optional[dict]:
    """
    Extract structured content using 3 strategies:
      1. GitHub Docs article body  (main article tag with sections)
      2. Heading + paragraph pairs (general docs structure)
      3. Bulk paragraph fallback
    """

    # ── Page title ─────────────────────────────────────────────────────────
    title = ""
    for sel in ["h1", ".article-title", "title"]:
        tag = soup.select_one(sel)
        if tag:
            title = tag.get_text(separator=" ", strip=True)
            title = title.split("|")[0].split("-")[0].strip()
            break

    qa_pairs: list[dict] = []

    # ── Strategy 1: GitHub Docs article structure ──────────────────────────
    # GitHub Docs wraps article content in <article> or .markdown-body
    article = soup.select_one(
        "article, .markdown-body, .article-body, "
        "[data-search='article-body'], main .prose"
    )

    if article:
        for heading in article.find_all(["h2", "h3"]):
            q_text = heading.get_text(separator=" ", strip=True)
            if not (10 < len(q_text) < 300):
                continue

            answer_parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3"]:
                    break
                if sibling.name in ["p", "ul", "ol", "blockquote", "div"]:
                    text = sibling.get_text(separator=" ", strip=True)
                    if text:
                        answer_parts.append(text)

            if answer_parts:
                full_answer = " ".join(answer_parts)[:2500]
                if len(full_answer) > 30:
                    qa_pairs.append({
                        "question": q_text,
                        "answer":   full_answer,
                    })

    # ── Strategy 2: Any main content area heading + paragraph ─────────────
    if not qa_pairs:
        main = (
            soup.select_one("main, #main-content, .main-content, #content")
            or soup.body
        )
        if main:
            for heading in main.find_all(["h2", "h3"]):
                q_text = heading.get_text(separator=" ", strip=True)
                if not (10 < len(q_text) < 300):
                    continue
                parts = []
                for sib in heading.find_next_siblings():
                    if sib.name in ["h2", "h3"]:
                        break
                    if sib.name in ["p", "ul", "ol"]:
                        t = sib.get_text(separator=" ", strip=True)
                        if t:
                            parts.append(t)
                if parts:
                    qa_pairs.append({
                        "question": q_text,
                        "answer":   " ".join(parts)[:2500],
                    })

    # ── Strategy 3: Bulk paragraph fallback ───────────────────────────────
    if not qa_pairs:
        main = (
            soup.select_one("main, article, .markdown-body, #content")
            or soup.body
        )
        if main:
            paragraphs = [
                p.get_text(separator=" ", strip=True)
                for p in main.find_all("p")
                if len(p.get_text(strip=True)) > 80
            ]
            if paragraphs:
                qa_pairs.append({
                    "question": title or "General information",
                    "answer":   " ".join(paragraphs)[:3000],
                })

    # Skip pages with no extractable content
    if not qa_pairs:
        return None

    # Skip pages where all answers are very short (likely nav-only pages)
    total_content = sum(len(qa["answer"]) for qa in qa_pairs)
    if total_content < MIN_CONTENT_LEN:
        return None

    category = detect_category(url)

    return {
        "doc_id":    make_doc_id(url),
        "url":       url,
        "title":     title,
        "category":  category,
        "source":    "GitHub Documentation",
        "qa_pairs":  qa_pairs,
        "full_text": f"{title}\n\n" + "\n\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in qa_pairs
        ),
    }


# ── Main crawler ───────────────────────────────────────────────────────────────


def crawl(max_pages: int = MAX_PAGES, delay_seconds: float = DELAY_SECONDS) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visited: set[str]  = set()
    queue:   list[str] = list(dict.fromkeys(SEED_URLS))
    saved    = 0
    skipped  = 0

    log.info("=" * 65)
    log.info("GitHub Docs Knowledge Base Crawler")
    log.info("Topics : Get Started | Repos | Billing | Auth | Orgs | Security")
    log.info("Max pages : %d  |  Output: %s/", max_pages, OUTPUT_DIR)
    log.info("=" * 65)

    while queue and len(visited) < max_pages:
        url = queue.pop(0)

        # Skip binary files immediately without fetching
        if is_binary_url(url):
            continue

        if url in visited:
            continue
        visited.add(url)

        log.info("[%3d visited | %3d saved]  %s", len(visited), saved, url)

        soup = fetch(url)
        if not soup:
            skipped += 1
            time.sleep(delay_seconds)
            continue

        # Discover new in-scope links
        for link in extract_links(soup, url):
            if link not in visited and link not in queue:
                queue.append(link)

        # Extract and save content
        doc = extract_content(soup, url)
        if doc and doc["qa_pairs"]:
            safe_cat = (
                doc["category"]
                .replace(" ", "_")
                .replace("&", "and")
                .replace("/", "-")
                .lower()
            )
            filename = f"{doc['doc_id']}_{safe_cat}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            saved += 1
            log.info(
                "  ✓ [%-22s]  '%s'  (%d sections)",
                doc["category"], doc["title"][:50], len(doc["qa_pairs"])
            )
        else:
            skipped += 1
            log.debug("  – No content: %s", url)

        time.sleep(delay_seconds)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Crawl complete")
    log.info("  Pages visited  : %d", len(visited))
    log.info("  Documents saved: %d  →  %s/", saved, OUTPUT_DIR)
    log.info("  Pages skipped  : %d", skipped)

    if saved < 50:
        log.warning(
            "Only %d docs saved (need 50+). "
            "Try increasing MAX_PAGES or adding more SEED_URLS.", saved
        )
    else:
        log.info("✅  Meets 50-document minimum (%d docs saved).", saved)


# ── ChromaDB integration helper ────────────────────────────────────────────────


def load_documents(directory: str = OUTPUT_DIR) -> list[dict]:
    """
    Load all scraped JSONs as LangChain-compatible document dicts.
    One dict per section — fine-grained chunks = better retrieval.

    Usage in your ingestion pipeline:
        from ingestion.scraper import load_documents
        docs = load_documents()
        # chunk + embed exactly as you do today with PDFs
    """
    if not os.path.exists(directory):
        log.error("Directory not found: %s — run crawl() first", directory)
        return []

    documents = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname), encoding="utf-8") as f:
            doc = json.load(f)

        for i, qa in enumerate(doc.get("qa_pairs", [])):
            documents.append({
                "page_content": f"Q: {qa['question']}\nA: {qa['answer']}",
                "metadata": {
                    "source":   doc["url"],
                    "title":    doc["title"],
                    "category": doc["category"],
                    "origin":   doc["source"],
                    "doc_id":   f"{doc['doc_id']}-{i}",
                },
            })
    return documents


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    crawl()

    docs = load_documents()
    print(f"\n📚  Total chunks ready for ChromaDB: {len(docs)}")

    if docs:
        print("\nSample chunk:")
        print(json.dumps(docs[0], indent=2))

        print("\nBreakdown by category:")
        for cat, count in Counter(
            d["metadata"]["category"] for d in docs
        ).most_common():
            print(f"  {cat:<30s}: {count} chunks")