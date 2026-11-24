
import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ddgs import DDGS

logger = logging.getLogger(__name__)

MAX_QUERIES = 2
MAX_RESULTS_PER_QUERY = 5
MAX_FINAL_RESULTS = 3
MIN_SNIPPET_LENGTH = 50


# Tier 1: official health organizations and research institutions.
TIER_1_TRUSTED = {
    "who.int",
    "cdc.gov",
    "nhs.uk",
    "medlineplus.gov",
    "nih.gov",
    "nia.nih.gov",
    "niddk.nih.gov",
    "heart.org",
    "cancer.gov",
    "fda.gov",
    "health.gov",
    "nutrition.gov",
    "usda.gov",
    "fao.org",
    "pubmed.ncbi.nlm.nih.gov",
    "usa.gov",
    "gov.uk",
    "europa.eu",
    "archives.gov",
    "loc.gov",
    "nationalarchives.gov.uk",
    "history.state.gov",
    "nasa.gov",
    "noaa.gov",
    "nsf.gov",
    "esa.int",
    "sec.gov",
    "federalreserve.gov",
    "imf.org",
    "worldbank.org",
    "sbp.org.pk",
}

# Tier 2: established medical institutions and health publishers.
TIER_2_TRUSTED = {
    "mayoclinic.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "health.harvard.edu",
    "msdmanuals.com",
    "healthline.com",
    "webmd.com",
    "medicalnewstoday.com",
    "verywellhealth.com",
    "goodrx.com",

    # General reference and historical sources
    "britannica.com",
    "history.com",

    # Scientific and technology research
    "nature.com",
    "link.springer.com",
    "arxiv.org",

    # Institutional technology research and reporting
    "news.mit.edu",
    "mitsloanedtech.mit.edu",
}

TIER_1_BANNED = {
    "youtube.com",
    "youtu.be",
    "wikipedia.org",
    "meetaugust.ai",
    "necolebitchie.com",
    "drberg.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "indianjadibooti.com",
    "biohunza.com",
    "cavemanorganics.pk",
    "vedikroots.com",
    "healthyly.in",
    "neurolaunch.com",
    "scienceinsights.org",
    "healthmodeonline.com",
    "facialadviser.com",
    "merakilane.com",
    "avzri.com",
    "wellrz.com",
    "mysymptom.org",
    "geniusgeo.blog",
    "acibademhealthpoint.com",
    "humnutrition.com",
}

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this",
    "from", "are", "was", "were", "has", "have",
    "been", "not", "but", "can", "may",
}

WORD_GROUPS = {
    "nail": {"nail", "nails", "fingernail", "fingernails"},
    "white_spot": {"white", "spot", "spots", "mark", "marks"},
    "dark_circle": {"dark", "circle", "circles", "eye", "eyes"},
    "mouth_ulcer": {"mouth", "ulcer", "ulcers", "canker", "sores"},
    "lip": {"lip", "lips", "chapped", "cracked"},
    "puffy_eye": {"puffy", "swollen", "swelling", "eyelid", "eyelids"},
    "deficiency": {"deficiency", "deficient", "lack", "low"},
}


def normalize_evidence_url(url: str) -> str:
    """Validate and normalize the basic structure of an evidence URL."""
    if not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlsplit(url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return ""

        if not parsed.hostname or parsed.username or parsed.password:
            return ""

        if parsed.port not in (None, 80, 443):
            return ""

        return urlunsplit((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query,
            "",
        ))

    except (ValueError, TypeError):
        return ""


def _matches_domain(hostname: str, domain: str) -> bool:
    """Match a domain itself or one of its subdomains."""
    hostname = hostname.lower().removeprefix("www.").rstrip(".")
    domain = domain.lower().removeprefix("www.").rstrip(".")

    return hostname == domain or hostname.endswith("." + domain)


def get_trust_score(hostname: str) -> int:
    """
    Rank approved publishers and research sources.

    This score measures publisher eligibility, not factual accuracy.
    """
    hostname = hostname.lower().removeprefix("www.").rstrip(".")

    if any(
        _matches_domain(hostname, domain)
        for domain in TIER_1_BANNED
    ):
        return -100

    if any(
        _matches_domain(hostname, domain)
        for domain in TIER_1_TRUSTED
    ):
        return 10

    if any(
        _matches_domain(hostname, domain)
        for domain in TIER_2_TRUSTED
    ):
        return 8

    return 0


def _normalize_word(word: str) -> str:
    """Apply basic word normalization for comparison."""
    word = word.lower()

    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"

    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]

    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]

    if len(word) > 4 and word.endswith("s"):
        return word[:-1]

    return word


def _get_tokens(text: str) -> set[str]:
    """Extract normalized keywords from English and Urdu text."""
    if not isinstance(text, str):
        return set()

    words = re.findall(r"[^\W_]{3,}", text.lower())

    return {
        _normalize_word(word)
        for word in words
        if word not in STOP_WORDS
    }


def _expand_tokens(tokens: set[str]) -> set[str]:
    """Expand selected health terms into related wording."""
    expanded = set(tokens)

    for group in WORD_GROUPS.values():
        normalized_group = {
            _normalize_word(word)
            for word in group
        }

        if tokens.intersection(normalized_group):
            expanded.update(normalized_group)

    return expanded


def _calculate_search_result_relevance(
    claim_text: str,
    title: str,
    snippet: str,
    search_context: str = "",
) -> float:
    """
    Calculate preliminary topical similarity.

    Search context helps match an Urdu claim to English search results.
    This score does not determine whether a source supports the claim.
    """
    claim_tokens = _expand_tokens(
        _get_tokens(claim_text) | _get_tokens(search_context)
    )

    title_tokens = _expand_tokens(_get_tokens(title))
    snippet_tokens = _expand_tokens(_get_tokens(snippet))

    if not claim_tokens:
        return 0.0

    title_overlap = len(claim_tokens & title_tokens) / len(claim_tokens)
    snippet_overlap = len(claim_tokens & snippet_tokens) / len(claim_tokens)

    score = (title_overlap * 0.60) + (snippet_overlap * 0.40)

    return round(min(score, 1.0), 3)


def _extract_relevant_passages(
    claim_text: str,
    page_text: str,
    max_passages: int = 2,
    min_sentence_length: int = 40,
) -> list[dict]:
    """
    Extract relevant sentences from verified webpage text.

    This identifies topical relevance only. It does not determine
    whether the evidence supports or contradicts the claim.
    """
    if not claim_text or not page_text:
        return []

    claim_tokens = _expand_tokens(_get_tokens(claim_text))

    if not claim_tokens:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", page_text)
    passages = []

    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip()

        if len(sentence) < min_sentence_length:
            continue

        sentence_tokens = _expand_tokens(_get_tokens(sentence))
        overlap = len(claim_tokens & sentence_tokens) / len(claim_tokens)

        if overlap <= 0:
            continue

        passages.append({
            "text": sentence,
            "relevance_score": round(overlap, 3),
        })

    passages.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    return passages[:max_passages]


def _clean_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove internal ranking fields from public evidence."""
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }



def retrieve_evidence_for_claim(
    claim: dict[str, Any],
) -> dict[str, Any]:
    """Retrieve, deduplicate, and rank candidate evidence sources."""

    import time

    claim_text = str(claim.get("text", "")).strip()

    if not claim_text:
        return {
            "evidence": [],
            "retrieval_status": "skipped",
        }

    raw_queries = claim.get("search_queries", [])

    if not isinstance(raw_queries, list):
        raw_queries = []

    queries = []
    seen_queries = set()

    for query in raw_queries:
        if not isinstance(query, str):
            continue

        query = query.strip()

        if not query:
            continue

        normalized_query = query.casefold()

        if normalized_query not in seen_queries:
            queries.append(query)
            seen_queries.add(normalized_query)

    # Always preserve the original claim as a search query.
    # Keep the generated search queries first.
    # Use the original claim only as a fallback if there is room.
    normalized_claim = claim_text.casefold()

    if len(queries) < MAX_QUERIES and normalized_claim not in seen_queries:
        queries.append(claim_text)

    if normalized_claim not in seen_queries:
        queries.append(claim_text)

    # Reserve the final search slot for the original claim.
    if len(queries) > MAX_QUERIES:
        if MAX_QUERIES <= 1:
            queries = [claim_text]
        else:
            queries = queries[:MAX_QUERIES - 1] + [claim_text]

    evidence_by_url = {}
    completed_searches = 0
    failed_searches = 0

    try:
        with DDGS() as search_client:

            for query in queries:
                results = None

                # Retry unexpected search errors up to three times.
                for attempt in range(1, 4):
                    try:
                        results = search_client.text(
                            query,
                            max_results=MAX_RESULTS_PER_QUERY,
                        )

                        break

                    except Exception as search_exc:
                        message = str(search_exc).lower()

                        # A normal no-results response is not a
                        # provider failure and does not need retrying.
                        if (
                            "no results" in message
                            or "no search results" in message
                        ):
                            logger.info(
                                "No results for query %r",
                                query,
                            )
                            break

                        logger.warning(
                            "Search attempt %s failed for %r: %s",
                            attempt,
                            query,
                            search_exc,
                        )

                        if attempt < 3:
                            time.sleep(attempt * 2)
                        else:
                            failed_searches += 1

                if results is None:
                    continue

                completed_searches += 1

                logger.info(
                    "DDGS returned %s raw results for query: %s",
                    len(results or []),
                    query,
                )

                for item in results or []:
                    if not isinstance(item, dict):
                        continue

                    url = normalize_evidence_url(
                        item.get("href", "")
                    )

                    if not url or url in evidence_by_url:
                        continue

                    hostname = urlsplit(url).hostname or ""
                    hostname = hostname.lower()

                    trust_score = get_trust_score(hostname)

                    if trust_score <= 0:
                        logger.info(
                            "Source rejected by trust filter: %s | "
                            "score: %s | query: %s",
                            hostname,
                            trust_score,
                            query,
                        )
                        continue

                    title = str(item.get("title", "")).strip()
                    snippet = str(item.get("body", "")).strip()

                    if len(snippet) < MIN_SNIPPET_LENGTH:
                        logger.info(
                            "Rejected short snippet: %s | length: %s",
                            url,
                            len(snippet),
                        )
                        continue

                    logger.info(
                        "Source passed trust and snippet filters: %s",
                        url,
                    )

                    relevance_score = (
                        _calculate_search_result_relevance(
                            claim_text=claim_text,
                            title=title,
                            snippet=snippet,
                            search_context=query,
                        )
                    )

                    logger.info(
                        "Search candidate: %s | trust: %.2f | "
                        "relevance: %.3f | query: %s",
                        url,
                        trust_score,
                        relevance_score,
                        query,
                    )

                    evidence_by_url[url] = {
                        "title": title or url,
                        "url": url,
                        "snippet": snippet,
                        "supports_claim": None,
                        "search_context": query,
                        "_trust_score": trust_score,
                        "_relevance_score": relevance_score,
                        "_hostname": hostname,
                    }

        # Rank by topical relevance first, then publisher trust.
        ordered = sorted(
            evidence_by_url.values(),
            key=lambda item: (
                item["_relevance_score"],
                item["_trust_score"],
            ),
            reverse=True,
        )

        final = [
            _clean_result(item)
            for item in ordered[:MAX_FINAL_RESULTS]
        ]

        if final:
            status = "success"
        elif completed_searches > 0:
            status = "no_sources_found"
        elif failed_searches > 0:
            status = "failed"
        else:
            status = "no_sources_found"

        return {
            "evidence": final,
            "retrieval_status": status,
        }

    except Exception:
        logger.exception(
            "Could not initialize evidence search for: %s",
            claim_text,
        )

        return {
            "evidence": [],
            "retrieval_status": "failed",
        }
