
import ipaddress
import logging
import re
import socket
from html import unescape
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from app.services.evidence_retrieval import normalize_evidence_url


logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 12
MAX_PAGE_BYTES = 1_000_000
MAX_TEXT_CHARS = 30_000
MIN_RELEVANCE_SCORE = 0.10


# Preliminary publisher candidates, not guarantees of accuracy.
TRUSTED_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "medlineplus.gov",
    "nhs.uk",
    "mayoclinic.org",
    "clevelandclinic.org",
    "cochranelibrary.com",
    "cochrane.org",
    "pubmed.ncbi.nlm.nih.gov",
    "medicalnewstoday.com",
    "verywellhealth.com",
    "healthline.com",
    "webmd.com",
    "health.com",
    "goodrx.com",
}


LOW_QUALITY_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "wikipedia.org",
}


# Limited synonym groups for the current health-focused terminology.

WORD_GROUPS = {
    # Health
    "nail": {"nail", "nails", "fingernail", "fingernails"},
    "white_spot": {"white", "spot", "spots", "mark", "marks"},
    "dark_circle": {"dark", "circle", "circles", "eye", "eyes"},
    "mouth_ulcer": {"mouth", "ulcer", "ulcers", "canker", "sores"},
    "lip": {"lip", "lips", "chapped", "cracked"},
    "puffy_eye": {"puffy", "swollen", "swelling", "eyelid", "eyelids"},
    "deficiency": {"deficiency", "deficient", "lack", "low"},

    # Technology
    "artificial_intelligence": {
        "artificial", "intelligence", "ai", "machine",
        "learning", "model", "algorithm", "algorithms",
    },
    "software": {
        "software", "application", "applications", "app",
        "program", "programming", "code", "system",
    },
    "cybersecurity": {
        "cybersecurity", "security", "privacy", "hacking",
        "malware", "phishing", "encryption",
    },

    # Science and research
    "research": {
        "research", "study", "studies", "experiment",
        "scientific", "evidence", "findings",
    },
    "climate": {
        "climate", "weather", "temperature", "warming",
        "rainfall", "drought", "carbon",
    },

    # History
    "historical_event": {
        "history", "historical", "event", "war",
        "battle", "empire", "century", "ancient",
    },

    # Finance and economics
    "finance": {
        "finance", "financial", "money", "bank",
        "economy", "economic", "inflation", "market",
    },

    # General education
    "education": {
        "education", "learning", "school", "student",
        "university", "teaching", "knowledge",
    },
}


STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from",
    "have", "has", "are", "was", "were", "you", "your",
    "can", "may", "not", "but", "into", "its", "their",
    "what", "when", "where", "which", "while", "does",
    "about", "after", "before", "because", "could", "would",
    "should", "they", "them", "there", "these", "those",
    "been", "being", "also", "than", "then", "such",
}


def _matches_domain(hostname: str, domain: str) -> bool:
    hostname = hostname.lower().removeprefix("www.").rstrip(".")
    domain = domain.lower().removeprefix("www.").rstrip(".")

    return hostname == domain or hostname.endswith("." + domain)


def _is_public_hostname(hostname: str) -> bool:
    """Reject IP addresses and DNS names resolving to non-public addresses."""

    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    hostname,
                    None,
                    type=socket.SOCK_STREAM,
                )
            ]
        except (OSError, ValueError):
            return False

    return bool(addresses) and all(
        address.is_global for address in addresses
    )


def _safe_url(url: str) -> str:
    """Validate a URL and reject local/private network destinations."""

    normalized = normalize_evidence_url(url)

    if not normalized:
        return ""

    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname

        if parsed.scheme not in {"http", "https"}:
            return ""

        if not hostname or parsed.username or parsed.password:
            return ""

        if parsed.port not in (None, 80, 443):
            return ""

        if not _is_public_hostname(hostname):
            return ""

        return normalized

    except ValueError:
        return ""


def _publisher_verified(hostname: str) -> bool:
    """
    Check whether the publisher is on the preliminary allowlist.

    This does not verify the accuracy of an individual article.
    """

    hostname = hostname.lower().removeprefix("www.").rstrip(".")

    if any(
        _matches_domain(hostname, domain)
        for domain in LOW_QUALITY_DOMAINS
    ):
        return False

    return any(
        _matches_domain(hostname, domain)
        for domain in TRUSTED_DOMAINS
    )

def _extract_page_text(soup: BeautifulSoup) -> str:
    # Remove elements that usually contain navigation or unrelated content.
    for element in soup.find_all([
        "nav",
        "footer",
        "header",
        "script",
        "style",
        "noscript",
        "form",
        "aside",
    ]):
        element.decompose()

    # Prefer the main article content.
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    # Extract meaningful content blocks separately.
    content_blocks = []

    for element in main_content.find_all([
        "h1",
        "h2",
        "h3",
        "p",
        "li",
        "blockquote",
    ]):
        text = element.get_text(" ", strip=True)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        # Ignore empty or very short blocks.
        if len(text) < 30:
            continue

        # Avoid duplicate blocks.
        if text not in content_blocks:
            content_blocks.append(text)

    # Fall back to general page text if semantic blocks are unavailable.
    if not content_blocks:
        text = main_content.get_text("\n", strip=True)
        text = unescape(text)
        text = re.sub(r"\n+", "\n", text).strip()
        return text[:MAX_TEXT_CHARS]

    return "\n".join(content_blocks)[:MAX_TEXT_CHARS]


def _normalize_token(token: str) -> str:
    """Normalize a word into a simple comparison form."""

    token = token.lower().strip()

    if len(token) > 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        token = token[:-1]

    return token


def _get_tokens(text: str) -> set[str]:
    """
    Extract normalized keywords from English and Urdu text.
    Unicode-aware tokenization allows Urdu words to be retained.
    """

    if not isinstance(text, str):
        return set()

    words = re.findall(r"[^\W_]{3,}", text.lower())

    return {
        _normalize_token(word)
        for word in words
        if word not in STOP_WORDS
    }


def _tokens(text: str) -> set[str]:
    """Compatibility helper using the shared token extraction."""

    return _get_tokens(text)


def _expand_tokens(tokens: set[str]) -> set[str]:
    """Add limited synonyms for common health terminology."""

    expanded = set(tokens)

    for group in WORD_GROUPS.values():
        normalized_group = {
            _normalize_token(word)
            for word in group
        }

        if tokens.intersection(normalized_group):
            expanded.update(normalized_group)

    return expanded


def _calculate_relevance(
    claim_text: str,
    title: str,
    snippet: str,
    page_text: str = "",
    search_context: str = "",
) -> tuple[float, bool, str]:
    """
    Calculate preliminary topical relevance.

    English search context helps compare Urdu claims with English pages.
    This does not determine whether a source supports or contradicts a claim.
    """

    claim_tokens = _expand_tokens(
        _get_tokens(claim_text)
        | _get_tokens(search_context)
    )

    title_tokens = _expand_tokens(_get_tokens(title))
    snippet_tokens = _expand_tokens(_get_tokens(snippet))
    page_tokens = _expand_tokens(_get_tokens(page_text))

    if not claim_tokens:
        return (
            0.0,
            False,
            "No searchable keywords were available.",
        )

    title_overlap = (
        len(claim_tokens & title_tokens) / len(claim_tokens)
    )

    snippet_overlap = (
        len(claim_tokens & snippet_tokens) / len(claim_tokens)
    )

    page_overlap = (
        len(claim_tokens & page_tokens) / len(claim_tokens)
    )

    score = (
        title_overlap * 0.25
        + snippet_overlap * 0.35
        + page_overlap * 0.40
    )

    score = round(min(score, 1.0), 3)
    relevant = score >= MIN_RELEVANCE_SCORE

    if relevant:
        reason = "The page contains keywords relevant to the claim."
    else:
        reason = "The page has insufficient topical keyword overlap."

    return score, relevant, reason


def _extract_relevant_passages(
    claim_text: str,
    page_text: str,
    max_passages: int = 2,
    min_sentence_length: int = 40,
    search_context: str = "",
) -> list[dict]:
    """
    Extract and rank claim-relevant sentences from webpage text.

    This is keyword-based passage retrieval, not factual verification.
    """

    if not claim_text or not page_text or max_passages <= 0:
        return []

    claim_tokens = (
            _get_tokens(claim_text)
            | _get_tokens(search_context)
    )

    if not claim_tokens:
        return []

    # Split the page into separate paragraphs.
    blocks = re.split(r"\n+", page_text)

    passages = []
    seen_sentences = set()

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        # Split each paragraph into sentence-like units.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            block,
        )

        for sentence in sentences:
            sentence = re.sub(
                r"\s+",
                " ",
                sentence,
            ).strip()

            # Ignore incomplete or very short sentences.
            if len(sentence) < min_sentence_length:
                continue

            # Ignore unfinished sentences.
            if sentence.endswith(":"):
                continue

            normalized = sentence.lower()

            if normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)

            sentence_tokens = _get_tokens(sentence)

            if not sentence_tokens:
                continue

            matched_keywords = claim_tokens & sentence_tokens

            if not matched_keywords:
                continue

            overlap = len(matched_keywords) / len(claim_tokens)

            # Ignore very weak keyword matches.
            if overlap < 0.15:
                continue
            passages.append({
                "text": sentence,
                "relevance_score": round(overlap, 3),
                "matched_keywords": sorted(matched_keywords),
            })

    passages.sort(
        key=lambda item: (
            item["relevance_score"],
            len(item["matched_keywords"]),
        ),
        reverse=True,
    )

    return passages[:max_passages]


def _snippet_matches_page(
    snippet: str,
    page_text: str,
) -> bool:
    """Check whether the search snippet overlaps with the retrieved page."""

    snippet_tokens = _tokens(snippet)
    page_tokens = _tokens(page_text)

    if len(snippet_tokens) < 3:
        return False

    overlap = len(snippet_tokens & page_tokens) / len(snippet_tokens)

    return overlap >= 0.50


def verify_evidence(
    claim_text: str,
    evidence: dict,
) -> dict:
    """
    Retrieve and inspect a candidate evidence webpage.

    This function checks publisher-list membership, page retrieval,
    and preliminary topical relevance. It does not determine whether
    the claim is true or false.
    """

    original_url = evidence.get("url", "")
    url = _safe_url(original_url)

    result = {
        "title": evidence.get("title") or original_url,
        "url": url or original_url,
        "snippet": evidence.get("snippet", ""),
        "supports_claim": None,
        "source_verified": False,
        "content_verified": False,
        "page_retrieved": False,
        "page_content_extracted": False,
        "snippet_matched": False,
        "relevant_to_claim": False,
        "relevance_score": 0.0,
        "relevance_reason": "Page has not been inspected.",
        "eligible_for_fact_check": False,
    }

    if not claim_text or not claim_text.strip() or not url:
        return result

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; VerityVisionAI/0.1)"
            )
        }

        current_url = url
        response = None
        chunks = []
        total_bytes = 0

        with requests.Session() as session:
            # Validate each redirect before requesting its destination.
            for _ in range(4):
                current_url = _safe_url(current_url)

                if not current_url:
                    return result

                response = session.get(
                    current_url,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=False,
                    stream=True,
                )

                if response.status_code not in (
                    301, 302, 303, 307, 308
                ):
                    break

                redirect = response.headers.get("Location")
                response.close()

                if not redirect:
                    return result

                current_url = urljoin(current_url, redirect)

            else:
                return result

            if response is None or response.status_code != 200:
                if response is not None:
                    response.close()
                return result

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" not in content_type:
                response.close()
                return result

            content_length = response.headers.get("Content-Length")

            if content_length:
                try:
                    if int(content_length) > MAX_PAGE_BYTES:
                        response.close()
                        return result
                except ValueError:
                    pass

            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue

                total_bytes += len(chunk)

                if total_bytes > MAX_PAGE_BYTES:
                    break

                chunks.append(chunk)

            response.close()

        if not chunks or total_bytes > MAX_PAGE_BYTES:
            return result

        html = b"".join(chunks)
        soup = BeautifulSoup(html, "html.parser")

        page_title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else result["title"]
        )

        page_text = _extract_page_text(soup)

        if len(page_text) < 200:
            result["relevance_reason"] = (
                "The retrieved page did not contain enough readable text."
            )
            return result

        final_url = _safe_url(current_url)

        if not final_url:
            return result

        hostname = urlsplit(final_url).hostname or ""

        result["url"] = final_url
        result["title"] = page_title or result["title"]
        result["page_retrieved"] = True
        result["page_content_extracted"] = True
        result["page_text"] = page_text
        result["page_text_length"] = len(page_text)

        # This means the page was retrieved and readable.
        # It does NOT mean the claim has been independently verified.
        result["content_verified"] = True

        result["source_verified"] = _publisher_verified(hostname)

        snippet = result["snippet"]

        result["snippet_matched"] = _snippet_matches_page(
            snippet,
            page_text,
        )

        score, relevant, reason = _calculate_relevance(
            claim_text=claim_text,
            title=result["title"],
            snippet=snippet,
            page_text=page_text,
            search_context=evidence.get("search_context", ""),
        )

        result["relevance_score"] = score
        result["relevant_to_claim"] = relevant
        result["relevance_reason"] = reason

        # Extract actual passages from the retrieved webpage.
        passages = _extract_relevant_passages(
            claim_text=claim_text,
            page_text=page_text,
            max_passages=3,
            search_context=evidence.get("search_context", ""),
        )

        result["passages"] = passages
        result["passage_count"] = len(passages)

        result["eligible_for_fact_check"] = (
            result["source_verified"]
            and result["page_retrieved"]
            and result["page_content_extracted"]
            and result["relevant_to_claim"]
            and bool(passages)
        )

        # Passage relevance does not establish factual support.
        result["supports_claim"] = None

        return result

    except requests.RequestException as exc:
        logger.info(
            "Could not fetch evidence URL %s: %s",
            url,
            exc,
        )
        return result

    except Exception:
        logger.exception(
            "Evidence verification failed for %s",
            url,
        )
        return result
