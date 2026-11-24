from functools import lru_cache
from faster_whisper import WhisperModel

# Change these two lines depending on where you run it:
#   Laptop (CPU):  "small", "cpu", "int8"
#   GPU machine:   "large-v3", "cuda", "float16"
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=4,
    )


def transcribe_audio(audio_path: str, language: str | None = "ur") -> dict:
    if language not in (None, "en", "ur"):
        raise ValueError("Supported language options are English, Urdu, or auto-detect.")

    model = get_whisper_model()

    prompt = "یہ اردو زبان کی گفتگو ہے۔" if language == "ur" else None

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,      # avoids cutting off quiet words
        },
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=prompt,         # keeps output in Urdu script
    )

    results = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text,
            })

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "text": " ".join(s["text"] for s in results),
        "segments": results,
    }