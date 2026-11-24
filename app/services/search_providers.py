"""
Search providers for evidence retrieval, with automatic fallback.

Every provider returns Tavily-shaped dicts, so the rest of
evidence_retrieval.py can stay unchanged:

    {"title": str, "url": str, "content": str, "published_date": str | None}

.env settings
    SEARCH_PROVIDERS   comma list, tried in order. Default: tavily,serper,europepmc
    TAVILY_API_KEY     free: 1,000 credits per month (resets monthly)
    SERPER_API_KEY     free: 2,500 queries once, no card (serper.dev)
    EUROPEPMC_EMAIL    optional, identifies your app to Europe PMC
    TAVILY_SEARCH_DEPTH  basic (default) or advanced
    SEARCH_CACHE_FILE  default search_cache.json
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 20
COOLDOWN_SECONDS = 3600  # a provider that ran out of quota is skipped for 1 hour

CACHE_FILE = Path(os.getenv("SEARCH_CACHE_FILE", "search_cache.json"))

_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")


class ProviderUnavailable(Exception):
    """One provider could not answer (no key, quota used up, network error)."""


class SearchUnavailable(Exception):
    """Every configured provider failed, so 'no results' would be misleading."""


def _providers() -> list[str]:
    raw = os.getenv("SEARCH_PROVIDERS", "tavily,serper,europepmc")
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _is_urdu_query(query: str) -> bool:
    return bool(_ARABIC_SCRIPT.search(query))


# --------------------------------------------------------------------------
# Tavily
# --------------------------------------------------------------------------

def _tavily(query: str, max_results: int) -> list[dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ProviderUnavailable("TAVILY_API_KEY is missing")

    payload = {
        "query": query,
        "topic": "general",
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    for attempt in range(2):
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ProviderUnavailable(f"tavily network error: {exc}") from exc

        if response.status_code in (429, 500, 502, 503) and attempt == 0:
            time.sleep(2)
            continue

        if response.status_code in (432, 433):
            raise ProviderUnavailable(
                f"tavily quota used up (status {response.status_code})"
            )

        if response.status_code != 200:
            raise ProviderUnavailable(
                f"tavily status={response.status_code} body={response.text[:200]}"
            )

        try:
            results = response.json().get("results", [])
        except ValueError as exc:
            raise ProviderUnavailable("tavily returned invalid JSON") from exc

        return [item for item in results if isinstance(item, dict)]

    raise ProviderUnavailable("tavily temporarily unavailable")


# --------------------------------------------------------------------------
# Serper (Google results). Urdu queries are sent with hl=ur, gl=pk.
# --------------------------------------------------------------------------

def _serper(query: str, max_results: int) -> list[dict[str, Any]]:
    api_key = os.getenv("SERPER_API_KEY")

    if not api_key:
        raise ProviderUnavailable("SERPER_API_KEY is missing")

    payload: dict[str, Any] = {"q": query, "num": max(1, min(max_results, 10))}

    if _is_urdu_query(query):
        payload.update({"hl": "ur", "gl": "pk"})
    else:
        payload.update({"hl": "en"})

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            json=payload,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ProviderUnavailable(f"serper network error: {exc}") from exc

    if response.status_code != 200:
        raise ProviderUnavailable(
            f"serper status={response.status_code} body={response.text[:200]}"
        )

    try:
        organic = response.json().get("organic", [])
    except ValueError as exc:
        raise ProviderUnavailable("serper returned invalid JSON") from exc

    return [
        {
            "title": item.get("title"),
            "url": item.get("link"),
            "content": item.get("snippet"),
            "published_date": item.get("date"),
        }
        for item in organic
        if isinstance(item, dict)
    ]


# --------------------------------------------------------------------------
# Europe PMC: free medical papers, no key. English queries only.
# --------------------------------------------------------------------------

def _europepmc(query: str, max_results: int) -> list[dict[str, Any]]:
    # Urdu queries cannot be answered by an English paper index.
    if _is_urdu_query(query):
        return []

    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": max(1, min(max_results, 10)),
    }

    email = os.getenv("EUROPEPMC_EMAIL")
    if email:
        params["email"] = email

    try:
        response = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ProviderUnavailable(f"europepmc network error: {exc}") from exc

    if response.status_code != 200:
        raise ProviderUnavailable(f"europepmc status={response.status_code}")

    try:
        records = response.json().get("resultList", {}).get("result", [])
    except ValueError as exc:
        raise ProviderUnavailable("europepmc returned invalid JSON") from exc

    results = []

    for record in records:
        if not isinstance(record, dict):
            continue

        abstract = re.sub(r"<[^>]+>", " ", str(record.get("abstractText") or ""))
        abstract = " ".join(abstract.split())
        source, record_id = record.get("source"), record.get("id")

        if not abstract or not source or not record_id:
            continue

        results.append(
            {
                "title": record.get("title"),
                "url": f"https://europepmc.org/article/{source}/{record_id}",
                "content": abstract,
                "published_date": record.get("firstPublicationDate")
                or record.get("pubYear"),
            }
        )

    return results


_PROVIDER_FUNCTIONS = {
    "tavily": _tavily,
    "serper": _serper,
    "europepmc": _europepmc,
}

# provider name -> time.time() after which it may be tried again
_disabled_until: dict[str, float] = {}


# --------------------------------------------------------------------------
# Cache (saves credits: the same query is never paid for twice)
# --------------------------------------------------------------------------

_cache: dict[str, Any] | None = None


def _load_cache() -> dict[str, Any]:
    global _cache

    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}

    return _cache


def _save_cache() -> None:
    try:
        CACHE_FILE.write_text(
            json.dumps(_cache or {}, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


# --------------------------------------------------------------------------
# Public function
# --------------------------------------------------------------------------

def search_raw(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search with the first provider that works.

    Returns Tavily-shaped dicts. Returns [] only when at least one provider
    answered and found nothing. Raises SearchUnavailable when every provider
    failed, so callers can report "search failed" instead of "no evidence".
    """
    query = " ".join(str(query or "").split())

    if not query:
        return []

    cache = _load_cache()
    cache_key = f"{max_results}|{query}"

    if cache_key in cache:
        return cache[cache_key]

    errors: list[str] = []
    answered = False

    for name in _providers():
        function = _PROVIDER_FUNCTIONS.get(name)

        if function is None:
            errors.append(f"{name}: unknown provider")
            continue

        if _disabled_until.get(name, 0) > time.time():
            errors.append(f"{name}: skipped (cooling down after an error)")
            continue

        try:
            results = function(query, max_results)
        except ProviderUnavailable as exc:
            logger.warning("Search provider failed: %s", exc)
            errors.append(str(exc))

            # Missing key or used-up quota will not fix itself within the hour.
            if "quota" in str(exc) or "missing" in str(exc) or "status=4" in str(exc):
                _disabled_until[name] = time.time() + COOLDOWN_SECONDS

            continue

        answered = True

        if results:
            # Only real results are cached, so a failure is never remembered.
            cache[cache_key] = results
            _save_cache()
            return results

    if not answered:
        raise SearchUnavailable("; ".join(errors) or "no search provider configured")

    return []