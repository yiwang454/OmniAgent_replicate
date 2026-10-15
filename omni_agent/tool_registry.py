"""Configuration-driven tool selection for OmniAgent ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from omni_agent.caption_cache import active_cached_audio_caption
from omni_agent.tool.Audio.audio_event_tools import (
    Audio_EventList,
    Audio_EventLocation,
    audio_qa,
)
from omni_agent.tool.Audio.audio_qa import (
    audio_ASR,
    audio_global_caption as _audio_global_caption,
)
from omni_agent.tool.Video.video_base import video_metadata
from omni_agent.tool.Video.video_qa import video_clip_qa, video_global_qa


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PROMPT_TEMPLATE_PATH = (
    REPO_ROOT / "configs" / "tool_ablation" / "prompts" / "base.txt"
)


@tool("audio_global_caption", description=_audio_global_caption.description)
def cached_audio_global_caption(video_path: str) -> dict[str, str]:
    """Registry adapter that serves the optional question-scoped cache."""
    cached = active_cached_audio_caption(video_path)
    if cached is not None:
        return {"answer": cached.response}
    return _audio_global_caption.invoke({"video_path": video_path})

# Keep the original build_agent ordering. Filtering this tuple changes only
# availability, not the relative order presented to the planner.
TOOL_REGISTRY = {
    "audio_global_caption": cached_audio_global_caption,
    "audio_qa": audio_qa,
    "video_global_qa": video_global_qa,
    "video_clip_qa": video_clip_qa,
    "video_metadata": video_metadata,
    "Audio_EventList": Audio_EventList,
    "Audio_EventLocation": Audio_EventLocation,
    "audio_ASR": audio_ASR,
}

TOOL_GROUPS = {
    "C": ("audio_global_caption",),
    "Q": ("audio_qa", "video_global_qa", "video_clip_qa"),
    "E": ("Audio_EventList", "Audio_EventLocation"),
    "A": ("audio_ASR",),
    "M": ("video_metadata",),
}


@dataclass(frozen=True)
class ToolSelectionConfig:
    experiment: str
    enabled_groups: tuple[str, ...]
    enabled_tool_names: tuple[str, ...]
    prompt_template_path: Path
    user_template: str
    config_path: Path

    @property
    def has_caption_tool(self) -> bool:
        return "audio_global_caption" in self.enabled_tool_names

    def tools(self) -> list[Any]:
        enabled = set(self.enabled_tool_names)
        return [tool for name, tool in TOOL_REGISTRY.items() if name in enabled]


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_tool_selection_config(path: Path | str) -> ToolSelectionConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported tool-selection config: {config_path}")

    experiment = str(payload.get("experiment") or "").strip()
    groups = tuple(str(value) for value in payload.get("enabled_groups") or ())
    if not experiment or not groups:
        raise ValueError(f"Config needs experiment and enabled_groups: {config_path}")
    unknown_groups = sorted(set(groups) - set(TOOL_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown tool groups in {config_path}: {unknown_groups}")
    if len(groups) != len(set(groups)):
        raise ValueError(f"Duplicate tool groups in {config_path}: {groups}")

    enabled = {
        tool_name
        for group in groups
        for tool_name in TOOL_GROUPS[group]
    }
    enabled_tool_names = tuple(name for name in TOOL_REGISTRY if name in enabled)
    declared_tools = tuple(str(value) for value in payload.get("enabled_tools") or ())
    if declared_tools and declared_tools != enabled_tool_names:
        raise ValueError(
            f"enabled_tools disagrees with enabled_groups in {config_path}: "
            f"expected={enabled_tool_names}, declared={declared_tools}"
        )

    prompt_value = str(payload.get("prompt_template") or "").strip()
    if not prompt_value:
        raise ValueError(f"Config has no prompt_template: {config_path}")
    prompt_path = _resolve_from_config(config_path, prompt_value)
    user_template = prompt_path.read_text(encoding="utf-8").strip()
    if not user_template:
        raise ValueError(f"Prompt template is empty: {prompt_path}")

    unavailable = set(TOOL_REGISTRY) - set(enabled_tool_names)
    leaked = sorted(name for name in unavailable if name in user_template)
    if leaked:
        raise ValueError(
            f"Prompt {prompt_path} mentions unavailable tools for {experiment}: {leaked}"
        )
    missing = sorted(name for name in enabled_tool_names if name not in user_template)
    if missing:
        raise ValueError(
            f"Prompt {prompt_path} omits enabled tools for {experiment}: {missing}"
        )

    return ToolSelectionConfig(
        experiment=experiment,
        enabled_groups=groups,
        enabled_tool_names=enabled_tool_names,
        prompt_template_path=prompt_path,
        user_template=user_template,
        config_path=config_path,
    )


def default_tools() -> list[Any]:
    """Return the unchanged full OmniAgent tool list."""
    return [
        _audio_global_caption,
        audio_qa,
        video_global_qa,
        video_clip_qa,
        video_metadata,
        Audio_EventList,
        Audio_EventLocation,
        audio_ASR,
    ]


def default_user_template() -> str:
    """Load the verbatim pre-ablation user prompt template."""
    return BASE_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
