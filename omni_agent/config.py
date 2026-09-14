"""Runtime configuration for OmniAgent.

Secrets are loaded only from this repository's untracked ``.env`` file (or
from already-exported process environment variables).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_environment() -> None:
    load_dotenv(_REPO_ROOT / ".env", override=False)


def _first_nonempty(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value.upper() != "EMPTY":
            return value
    return default


_load_environment()

# ------------------ Main agent configuration ------------------ #
# The ELM setup in react-agent-avqa-optimize deliberately uses the OpenAI SDK's
# default endpoint. Do not inject a DeepSeek or other OpenAI-compatible URL.
OPENAI_API_KEY = _first_nonempty("ELM_API_KEY", "OPENAI_API_KEY")
BRAIN_MODEL = os.getenv("BRAIN_MODEL", "o3")

# ------------------ Multimodal model configuration ------------- #
YOUR_API_KEY_GEMINI = _first_nonempty("GEMINI_API_KEY")
GEMINI_BASE_URL = _first_nonempty(
    "GEMINI_BASE_URL", default="https://api.apiplus.org"
).rstrip("/")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "180"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "6"))
GEMINI_RETRY_DELAY_S = float(os.getenv("GEMINI_RETRY_DELAY_S", "5"))
GEMINI_RETRY_MAX_DELAY_S = float(os.getenv("GEMINI_RETRY_MAX_DELAY_S", "60"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))

# All high-level audio/video perception tools use Gemini 2.5 Flash.
VIDEO_TOOL = "GEMINI"
ASR_GC_TOOL = "GEMINI"
LOCATION_TOOL = "GEMINI"
