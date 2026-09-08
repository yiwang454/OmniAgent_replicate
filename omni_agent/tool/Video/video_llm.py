"""Video MLLM adapter backed exclusively by Gemini 2.5 Flash."""

from omni_agent.gemini_api import call_gemini_with_media


def video_llm_gemini(video_path: str, text_block: str, fps: float = 2) -> str:
    return call_gemini_with_media(video_path, text_block, fps=fps)


# Compatibility for external code that imported the old generic name.
video_llm = video_llm_gemini
