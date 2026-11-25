
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from app.services.transcription import transcribe_audio


FFMPEG_PATH = (
    os.getenv("FFMPEG_PATH")
    or imageio_ffmpeg.get_ffmpeg_exe()
)


def extract_audio_and_transcribe(video_path: str) -> dict:
    """Extract audio from a video and transcribe it."""

    if not Path(FFMPEG_PATH).is_file():
        raise RuntimeError("FFmpeg executable was not found.")

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as audio_file:
        audio_path = audio_file.name

    try:
        subprocess.run(
            [
                FFMPEG_PATH,
                "-nostdin",
                "-v", "error",
                "-i", video_path,
                "-map", "0:a:0",
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-t", "600",
                "-y",
                audio_path,
            ],
            check=True,
            timeout=180,
            capture_output=True,
            text=True,
        )

        return transcribe_audio(audio_path, language="en")

    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "Could not extract audio. Check that the video "
            "is valid and contains an audio track."
        ) from exc

    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio extraction timed out.") from exc

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
