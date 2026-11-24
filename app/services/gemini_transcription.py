
import json
import logging
import math
import os
import random
import re
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

MODEL = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.5-flash-lite")

VALID_MODES = {"auto", "urdu", "english", "roman_urdu"}

MAX_ATTEMPTS = 4

# Retry only temporary API/service failures.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

BASE_RETRY_DELAY = 2
MAX_RETRY_DELAY = 15


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _get_status_code(exc: Exception):
    """Extract an HTTP status code from a Gemini SDK exception."""

    code = getattr(exc, "code", None)

    if code is None:
        code = getattr(exc, "status_code", None)

    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _is_retryable_error(exc: Exception) -> bool:
    """Determine whether an exception is likely temporary."""

    status_code = _get_status_code(exc)

    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES

    # Gemini SDK may expose temporary failures through exception classes.
    exception_name = type(exc).__name__.lower()

    if "servererror" in exception_name:
        return True

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    return False


def _retry_delay(attempt: int) -> float:
    """Exponential backoff with random jitter."""

    delay = min(
        BASE_RETRY_DELAY * (2 ** attempt),
        MAX_RETRY_DELAY
    )

    return delay + random.uniform(0, 1)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BASE_PROMPT = """Transcribe this audio exactly as spoken, split into short
segments of one sentence or phrase each (about 3 to 8 seconds).

Return ONLY a JSON object, with no markdown and no extra text, like:
{"language": "ur", "segments": [{"start": 0.0, "end": 5.2, "text": "..."}]}

General rules:
- "language" is "ur" (Urdu/Hindi), "en" (English), or "mixed".
- "start" and "end" are in seconds (numbers).
- Transcribe every spoken word, including numbered items like
  "first", "second", "پہلی", "دوسری", "چوتھی". Do not skip short phrases.
- Do not write filler sounds such as "uh", "um", "اہ", "ام".
- Do not summarize, translate, or add anything.
- If a word is unclear, write [?] instead of guessing.
- Segments should cover the audio from the first spoken word to the last.
- Output valid JSON only. Write every character as a normal character,
  never as a broken or partial \\u escape.
"""

MODE_RULES = {
    "auto": """
Script rules (auto mode):
- Detect the spoken language.
- Urdu speech: write in Urdu script with correct, standard spelling.
  English words spoken inside Urdu sentences (like "mental health") are
  written in Urdu script.
- English speech: write in normal English with correct spelling and punctuation.
""",

    "urdu": """
Script rules (Urdu mode):
- Write everything in Urdu script with correct, standard Urdu spelling.
- English words spoken in the audio are written in Urdu script.
- The pronoun "you" is always آپ (with madda), never اپ.
- Write آج (not اج), ذہن (mind, not ذہین), ورزش (exercise), ہوگا (not ہوے گا).
- Use the correct health and medical words when the topic is health,
  for example: درد، بخار، نیند، کمی، وٹامن، صحت.
""",

    "english": """
Script rules (English mode):
- Write everything in English (Latin letters) with correct spelling
  and punctuation.
- If a speaker says Urdu words, write them in Roman Urdu.
""",

    "roman_urdu": """
Script rules (Roman Urdu mode):
- Write the Urdu/Hindi speech in Roman Urdu, meaning Urdu written with
  English letters, the way people type it in chat. Do NOT use Urdu script.
- Use simple, consistent spellings across the whole transcript:
  aap, hai, hain, nahi, kya, kyun, mein, ka, ki, ke, ko, se, aur, bohat,
  theek, warzish, neend, sehat, dard.
- Write Urdu words as Urdu even when they sound like English words
  (for example "neend" means sleep, never "need").
- Keep words that are really spoken in English in normal English spelling
  (e.g. mental health, vitamin, zinc, deficiency).
- Use lowercase except for names and the start of sentences.
"""
}


# ---------------------------------------------------------------------------
# Urdu spelling safety net
# ---------------------------------------------------------------------------

_ARABIC = r"\u0600-\u06FF"

_STANDALONE_AAP = re.compile(
    rf"(?<![{_ARABIC}])اپ(?![{_ARABIC}])"
)

_STANDALONE_AAJ = re.compile(
    rf"(?<![{_ARABIC}])اج(?![{_ARABIC}])"
)

_FILLER = re.compile(
    rf"(?<![{_ARABIC}])(اہ|ام)(?![{_ARABIC}])\s*"
)

_REPLACEMENTS = {
    "ورزشی کرتے": "ورزش کرتے",
    "ہوے گا": "ہوگا",
}


def fix_common_spelling(text: str) -> str:
    """Apply limited Urdu spelling corrections."""

    text = _STANDALONE_AAP.sub("آپ", text)
    text = _STANDALONE_AAJ.sub("آج", text)
    text = _FILLER.sub("", text)

    for wrong, right in _REPLACEMENTS.items():
        text = text.replace(wrong, right)

    return re.sub(r"\s{2,}", " ", text).strip()


def build_prompt(mode: str) -> str:
    if mode not in MODE_RULES:
        raise ValueError(f"Unsupported transcription mode: {mode}")

    return BASE_PROMPT + MODE_RULES[mode]


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------

def _normalize_language(value) -> str | None:
    if not value:
        return None

    value = str(value).strip().lower()

    if value in ("ur", "urdu", "hi", "hindi"):
        return "ur"

    if value in ("en", "english"):
        return "en"

    if value == "mixed":
        return "mixed"

    return None


def _build_segment(start, end, text) -> dict | None:
    """Validate and normalize one transcript segment."""

    text = fix_common_spelling(str(text))

    if not text:
        return None

    try:
        start = float(start)
        end = float(end)

        if not math.isfinite(start) or not math.isfinite(end):
            return None

        if start < 0 or end <= start:
            return None

        return {
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text
        }

    except (TypeError, ValueError):
        return None


def _strip_fences(raw: str) -> str:
    """Remove optional Markdown JSON fences."""

    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(r"\s*```$", "", raw)

    return raw.strip()


def _parse(raw: str):
    """Strictly parse and validate Gemini's JSON response."""

    data = json.loads(_strip_fences(raw))

    if isinstance(data, list):
        language = None
        items = data

    elif isinstance(data, dict):
        language = data.get("language")
        items = data.get("segments", [])

    else:
        raise ValueError("Unexpected JSON response structure.")

    if not isinstance(items, list):
        raise ValueError("'segments' must be a list.")

    segments = []

    for item in items:
        if not isinstance(item, dict):
            continue

        segment = _build_segment(
            item.get("start"),
            item.get("end"),
            item.get("text", "")
        )

        if segment:
            segments.append(segment)

    if not segments:
        raise ValueError("No valid segments found in Gemini response.")

    return _normalize_language(language), segments


# ---------------------------------------------------------------------------
# Partial response recovery
# ---------------------------------------------------------------------------

_SEGMENT_RE = re.compile(
    r'"start"\s*:\s*([\d.]+)\s*,\s*'
    r'"end"\s*:\s*([\d.]+)\s*,\s*'
    r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL
)

_BAD_UNICODE = re.compile(
    r"\\u(?![0-9a-fA-F]{4})[0-9a-fA-F]*"
)


def _decode_text(escaped: str) -> str:
    """Decode a JSON string, handling broken Unicode escapes."""

    escaped = _BAD_UNICODE.sub("[?]", escaped)

    try:
        return json.loads(f'"{escaped}"')

    except json.JSONDecodeError:
        return escaped


def _salvage(raw: str):
    """Recover valid segments from a partially malformed JSON response."""

    segments = []

    for start, end, text in _SEGMENT_RE.findall(raw):

        segment = _build_segment(
            start,
            end,
            _decode_text(text)
        )

        if segment:
            segments.append(segment)

    language_match = re.search(
        r'"language"\s*:\s*"([^"]+)"',
        raw
    )

    language = (
        _normalize_language(language_match.group(1))
        if language_match
        else None
    )

    return language, segments


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def _generate(prompt: str, uploaded, temperature: float) -> str:

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, uploaded],
        config={
            "temperature": temperature,
            "response_mime_type": "application/json"
        }
    )

    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Retry and response handling
# ---------------------------------------------------------------------------

def _transcribe_with_retry(prompt: str, uploaded):
    """
    Transcribe audio with controlled retries.

    Retries:
    - Temporary Gemini API errors.
    - Empty responses.
    - Invalid or malformed JSON.

    Keeps valid partial segments if a later attempt fails.
    """

    last_error = None

    best_language = None
    best_segments = []

    for attempt in range(MAX_ATTEMPTS):

        temperature = 0 if attempt == 0 else 0.3

        try:
            raw = _generate(
                prompt,
                uploaded,
                temperature
            )

        except Exception as exc:

            last_error = exc

            if not _is_retryable_error(exc):

                logger.error(
                    "Non-retryable Gemini transcription error: "
                    "model=%s error_type=%s",
                    MODEL,
                    type(exc).__name__
                )

                break

            logger.warning(
                "Temporary Gemini transcription error: "
                "model=%s attempt=%s/%s status=%s error_type=%s",
                MODEL,
                attempt + 1,
                MAX_ATTEMPTS,
                _get_status_code(exc),
                type(exc).__name__
            )

            if attempt < MAX_ATTEMPTS - 1:
                delay = _retry_delay(attempt)

                logger.info(
                    "Retrying transcription in %.2f seconds.",
                    delay
                )

                time.sleep(delay)

            continue

        if not raw:

            last_error = ValueError(
                "Gemini returned an empty response."
            )

            logger.warning(
                "Gemini returned an empty transcription response: "
                "attempt=%s/%s",
                attempt + 1,
                MAX_ATTEMPTS
            )

        else:

            try:
                language, segments = _parse(raw)

                logger.info(
                    "Gemini transcription successful: "
                    "model=%s segments=%s attempt=%s",
                    MODEL,
                    len(segments),
                    attempt + 1
                )

                return language, segments

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError,
                AttributeError
            ) as exc:

                last_error = exc

                language, segments = _salvage(raw)

                if len(segments) > len(best_segments):

                    best_language = language
                    best_segments = segments

                logger.warning(
                    "Gemini transcription response validation failed: "
                    "attempt=%s/%s recovered_segments=%s error=%s",
                    attempt + 1,
                    MAX_ATTEMPTS,
                    len(segments),
                    str(exc)[:200]
                )

        if attempt < MAX_ATTEMPTS - 1:

            delay = _retry_delay(attempt)

            logger.info(
                "Retrying transcription response in %.2f seconds.",
                delay
            )

            time.sleep(delay)

    # Return recovered segments if strict parsing failed but
    # some valid transcript segments were recovered.
    if best_segments:

        logger.warning(
            "Returning partial recovered transcription: segments=%s",
            len(best_segments)
        )

        return best_language, best_segments

    raise RuntimeError(
        f"Gemini transcription failed: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public transcription function
# ---------------------------------------------------------------------------

def transcribe_with_gemini(
    audio_path: str,
    mode: str = "auto"
) -> dict:
    """
    Transcribe an audio file using Gemini.

    Modes:
        auto       Detect language automatically.
        urdu       Write speech in Urdu script.
        english    Write speech in English/Roman Urdu.
        roman_urdu Write Urdu speech in Roman Urdu.
    """

    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of: {', '.join(sorted(VALID_MODES))}"
        )

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file does not exist: {audio_path}"
        )

    uploaded = None

    try:
        logger.info(
            "Uploading audio for Gemini transcription: model=%s mode=%s",
            MODEL,
            mode
        )

        uploaded = client.files.upload(file=audio_path)

        language, segments = _transcribe_with_retry(
            build_prompt(mode),
            uploaded
        )

    finally:

        if uploaded is not None:

            try:
                client.files.delete(name=uploaded.name)

            except Exception as exc:

                logger.warning(
                    "Could not delete temporary Gemini upload: "
                    "error_type=%s",
                    type(exc).__name__
                )

    if mode == "english":
        fallback_language = "en"
    else:
        fallback_language = "ur"

    transcript_text = " ".join(
        segment["text"] for segment in segments
    )

    return {
        "language": language or fallback_language,
        "mode": mode,
        "text": transcript_text,
        "segments": segments
    }
