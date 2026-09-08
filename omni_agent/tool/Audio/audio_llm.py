"""Audio MLLM adapter backed exclusively by Gemini 2.5 Flash."""

from __future__ import annotations

import os
import subprocess
import tempfile

from omni_agent.gemini_api import call_gemini_with_media


def _existing_audio_path(video_path: str) -> str | None:
    audio_path = video_path.replace(".mp4", ".wav").replace("videos", "audios")
    return audio_path if audio_path != video_path and os.path.exists(audio_path) else None


def audio_llm_gemini(
    video_path: str,
    question: str,
    system_prompt: str | None = None,
) -> str:
    audio_path = _existing_audio_path(video_path)
    if audio_path:
        return call_gemini_with_media(
            audio_path,
            question,
            system_prompt=system_prompt,
        )

    wav_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vn",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return call_gemini_with_media(
            wav_path,
            question,
            system_prompt=system_prompt,
        )
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


# Compatibility for external code that imported the old generic name.
audio_llm = audio_llm_gemini
