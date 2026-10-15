from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from omni_agent.caption_cache import (
    AUDIO_GLOBAL_CAPTION_PROMPT,
    activate_caption_cache,
    load_cached_audio_caption,
)
from omni_agent.tracing import RolloutTraceRecorder, activate_trace_recorder
from omni_agent.tool_registry import (
    TOOL_GROUPS,
    TOOL_REGISTRY,
    default_user_template,
    cached_audio_global_caption,
    load_tool_selection_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("omniagent_main_ablation", REPO_ROOT / "main.py")
assert SPEC is not None and SPEC.loader is not None
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)


EXPECTED_GROUPS = {
    "minus_a": ("C", "Q", "E", "M"),
    "minus_e": ("C", "Q", "A", "M"),
    "minus_a_e": ("C", "Q", "M"),
    "minus_c": ("Q", "E", "A", "M"),
    "minus_q": ("C", "E", "A", "M"),
    "minus_m": ("C", "Q", "E", "A"),
    "minimal": ("C", "Q"),
}


class ToolAblationTests(unittest.TestCase):
    def test_all_experiment_configs_expose_exact_group_union(self):
        for experiment, expected_groups in EXPECTED_GROUPS.items():
            config = load_tool_selection_config(
                REPO_ROOT / "configs" / "tool_ablation" / f"{experiment}.json"
            )
            self.assertEqual(config.enabled_groups, expected_groups)
            expected_names = {
                name for group in expected_groups for name in TOOL_GROUPS[group]
            }
            self.assertEqual(set(config.enabled_tool_names), expected_names)
            self.assertEqual(
                [tool.name for tool in config.tools()],
                [name for name in TOOL_REGISTRY if name in expected_names],
            )
            for unavailable in set(TOOL_REGISTRY) - expected_names:
                self.assertNotIn(unavailable, config.user_template)

    def test_prompt_variants_preserve_non_tool_sections(self):
        base = default_user_template()
        prefix, base_tail = base.split("    Tool usage guidelines for this task:\n", 1)
        _, suffix = base_tail.split("\n    Remember:\n", 1)
        for experiment in EXPECTED_GROUPS:
            config = load_tool_selection_config(
                REPO_ROOT / "configs" / "tool_ablation" / f"{experiment}.json"
            )
            variant_prefix, variant_tail = config.user_template.split(
                "    Tool usage guidelines for this task:\n", 1
            )
            _, variant_suffix = variant_tail.split("\n    Remember:\n", 1)
            self.assertEqual(variant_prefix, prefix)
            self.assertEqual(variant_suffix, suffix)

    def test_cache_keeps_original_generation_usage_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(
                json.dumps({"total_wall_seconds": 12.5}), encoding="utf-8"
            )
            usage = {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 3,
            }
            (root / "q1.json").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "response": "caption",
                        "caption_prompt": AUDIO_GLOBAL_CAPTION_PROMPT,
                        "source_results_file": str(source),
                        "source_token_usage": usage,
                        "source_backend": "gemini_audio_global_caption",
                    }
                ),
                encoding="utf-8",
            )
            cached = load_cached_audio_caption(root, "q1")
        self.assertEqual(cached.response, "caption")
        self.assertEqual(cached.metadata["caption_cache_source_token_usage"], usage)
        self.assertEqual(cached.metadata["caption_cache_source_latency_seconds"], 12.5)

    def test_registry_caption_adapter_returns_cache_without_live_tool_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "q1.json").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "response": "cached caption",
                        "caption_prompt": AUDIO_GLOBAL_CAPTION_PROMPT,
                    }
                ),
                encoding="utf-8",
            )
            recorder = RolloutTraceRecorder()
            with activate_trace_recorder(recorder), activate_caption_cache(root, "q1"):
                result = cached_audio_global_caption.invoke({"video_path": "alias.mp4"})
        self.assertEqual(result, {"answer": "cached caption"})
        self.assertEqual(len(recorder.caption_cache_calls), 1)
        self.assertTrue(recorder.caption_cache_calls[0]["caption_cache_hit"])

    def test_sepav_alias_uses_video_and_audio_from_media_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source_sil.mp4"
            audio = root / "source_audio.wav"
            video.write_bytes(b"video")
            audio.write_bytes(b"audio")
            cut = {
                "id": "q1",
                "supervisions": [
                    {
                        "id": "q1",
                        "text": "Question?",
                        "custom": {
                            "options": ["A. a", "B. b"],
                            "video_path": "/wrong/combined.mp4",
                        },
                    }
                ],
            }
            media_cut = {
                "id": "q1",
                "supervisions": [
                    {
                        "id": "q1",
                        "custom": {
                            "video_path": str(video),
                            "audio_path": str(audio),
                        },
                    }
                ],
            }
            payload = main.build_input(
                cut, media_cut=media_cut, media_alias_dir=root / "aliases"
            )
            alias = Path(payload["video_path"])
            self.assertEqual(alias.resolve(), video.resolve())
            self.assertEqual(alias.with_suffix(".wav").resolve(), audio.resolve())


if __name__ == "__main__":
    unittest.main()
