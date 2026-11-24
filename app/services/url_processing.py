import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from app.services.transcription import transcribe_audio
from app.services.video_processing import FFMPEG_PATH
from app.services.gemini_transcription import transcribe_with_gemini


ALLOWED_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "fb.watch",
    "vimeo.com",
    "dailymotion.com",
    "x.com",
    "twitter.com",
}

MAX_VIDEO_DURATION = 600


def is_supported_url(url: str) -> bool:
    """Allow HTTPS URLs from supported video platforms."""

    parsed = urlparse(url)

    if parsed.scheme != "https" or not parsed.hostname:
        return False

    hostname = parsed.hostname.lower().rstrip(".")

    return any(
        hostname == domain or hostname.endswith("." + domain)
        for domain in ALLOWED_DOMAINS
    )


def process_video_url(url: str, mode: str = "auto") -> dict:
    """Extract audio from a public video URL and transcribe it.

    language: "ur" (Urdu), "en" (English), or None (auto-detect).
    """

    if not is_supported_url(url):
        raise ValueError("Unsupported or invalid video URL.")

    if not Path(FFMPEG_PATH).is_file():
        raise RuntimeError("FFmpeg executable was not found. Check FFMPEG_PATH.")

    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "socket_timeout": 10,
        "retries": 1,
        "quiet": True,
        "no_warnings": True,
    }

    audio_path = None

    try:
        # Stage 1: Resolve video metadata and audio stream.
        stage_started = time.perf_counter()

        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)

        print("URL resolution:", round(time.perf_counter() - stage_started, 2), "seconds")

        if not info or not info.get("url"):
            raise ValueError("No accessible audio stream was found.")

        duration = info.get("duration")

        if duration is not None and duration > MAX_VIDEO_DURATION:
            raise ValueError("Video must be 10 minutes or shorter.")

        stream_url = info["url"]

        headers = info.get("http_headers", {})
        header_string = "".join(f"{key}: {value}\r\n" for key, value in headers.items())

        # Stage 2: Extract mono, 16 kHz audio to a temporary WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = audio_file.name

        command = [FFMPEG_PATH, "-nostdin", "-v", "error"]

        if header_string:
            command.extend(["-headers", header_string])

        command.extend(
            [
                "-i", stream_url,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-t", str(MAX_VIDEO_DURATION),
                "-c:a", "pcm_s16le",
                "-y",
                audio_path,
            ]
        )

        stage_started = time.perf_counter()

        subprocess.run(
            command,
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )

        print("Audio extraction:", round(time.perf_counter() - stage_started, 2), "seconds")

        # Stage 3: Transcribe (Gemini first, local Whisper as fallback).
        stage_started = time.perf_counter()

        transcript = transcribe_with_gemini(audio_path, mode=mode)

        print("Transcription:", round(time.perf_counter() - stage_started, 2), "seconds")

        return {
            "filename": Path(info.get("title") or "video").name,
            "transcript": transcript,
        }

    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(
            "Could not access the video audio stream. "
            "The video may be private or restricted."
        ) from exc

    except subprocess.CalledProcessError as exc:
        raise ValueError("FFmpeg could not extract audio from this video.") from exc

    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio extraction timed out.") from exc

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)