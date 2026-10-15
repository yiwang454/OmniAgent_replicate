from __future__ import annotations

import base64
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_agent import config
from omni_agent import gemini_api
from omni_agent import latency
from omni_agent.tracing import RolloutTraceRecorder, activate_trace_recorder


class _FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "internal", "thought": True},
                            {"text": "answer"},
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 12,
                "thoughtsTokenCount": 8,
                "totalTokenCount": 120,
                "promptTokensDetails": [
                    {"modality": "VIDEO", "tokenCount": 90},
                    {"modality": "TEXT", "tokenCount": 10},
                ],
            },
        }


class _FakeErrorResponse:
    ok = False
    status_code = 400
    text = (
        '{"error":{"message":"contents[0].parts[1].data: '
        "required oneof field 'data' must have one initialized field\"}}"
    )

    def json(self):
        return {}


class ApiConfigurationTests(unittest.TestCase):
    def test_explicit_env_file_must_exist(self):
        with (
            patch.dict(
                "os.environ",
                {"OMNIAGENT_ENV_FILE": "/definitely/missing/omniagent.env"},
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "OMNIAGENT_ENV_FILE does not exist"),
        ):
            config._load_environment()

    def test_explicit_env_file_overrides_inherited_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("ELM_API_KEY=explicit-key\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "OMNIAGENT_ENV_FILE": str(env_path),
                    "ELM_API_KEY": "old-key",
                },
                clear=False,
            ):
                config._load_environment()
                self.assertEqual(config.os.environ["ELM_API_KEY"], "explicit-key")

    def test_all_multimodal_routes_are_fixed_to_gemini_25_flash(self):
        self.assertEqual(config.GEMINI_MODEL, "gemini-2.5-flash")
        self.assertEqual(config.VIDEO_TOOL, "GEMINI")
        self.assertEqual(config.ASR_GC_TOOL, "GEMINI")
        self.assertEqual(config.LOCATION_TOOL, "GEMINI")

    def test_gemini_request_matches_legacy_bearer_endpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            media.write(b"test media")
            media.flush()
            with (
                patch.object(gemini_api, "YOUR_API_KEY_GEMINI", "test-key"),
                patch.object(gemini_api, "GEMINI_BASE_URL", "https://gemini.example"),
                patch.object(gemini_api.requests, "post", return_value=_FakeResponse()) as post,
            ):
                answer = gemini_api.call_gemini_with_media(
                    media.name,
                    "describe",
                    system_prompt="be precise",
                    fps=5,
                )

        self.assertEqual(answer, "answer")
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://gemini.example/v1beta/models/gemini-2.5-flash:generateContent",
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        media_part = kwargs["json"]["contents"][0]["parts"][0]
        self.assertEqual(media_part["videoMetadata"], {"fps": 5})
        self.assertEqual(
            base64.b64decode(media_part["inlineData"]["data"]),
            b"test media",
        )
        self.assertEqual(
            kwargs["json"]["systemInstruction"],
            {"parts": [{"text": "be precise"}]},
        )

    def test_part_oneof_gateway_error_retries_with_text_first(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            media.write(b"test media")
            media.flush()
            with (
                patch.object(gemini_api, "YOUR_API_KEY_GEMINI", "test-key"),
                patch.object(gemini_api, "GEMINI_BASE_URL", "https://gemini.example"),
                patch.object(gemini_api, "GEMINI_MAX_RETRIES", 2),
                patch.object(gemini_api, "GEMINI_RETRY_DELAY_S", 0),
                patch.object(
                    gemini_api.requests,
                    "post",
                    side_effect=[_FakeErrorResponse(), _FakeResponse()],
                ) as post,
            ):
                answer = gemini_api.call_gemini_with_media(media.name, "describe", fps=2)

        self.assertEqual(answer, "answer")
        self.assertEqual(post.call_count, 2)
        first_parts = post.call_args_list[0].kwargs["json"]["contents"][0]["parts"]
        second_parts = post.call_args_list[1].kwargs["json"]["contents"][0]["parts"]
        self.assertIn("inlineData", first_parts[0])
        self.assertEqual(first_parts[1], {"text": "describe"})
        self.assertEqual(second_parts[0], {"text": "describe"})
        self.assertIn("inlineData", second_parts[1])

    def test_429_error_is_retried(self):
        error = _FakeErrorResponse()
        error.status_code = 429
        error.text = '{"error":{"message":"rate limit exceeded"}}'
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            media.write(b"test media")
            media.flush()
            with (
                patch.object(gemini_api, "YOUR_API_KEY_GEMINI", "test-key"),
                patch.object(gemini_api, "GEMINI_BASE_URL", "https://gemini.example"),
                patch.object(gemini_api, "GEMINI_MAX_RETRIES", 6),
                patch.object(gemini_api, "GEMINI_RETRY_DELAY_S", 0),
                patch.object(
                    gemini_api.requests,
                    "post",
                    side_effect=[error, _FakeResponse()],
                ) as post,
            ):
                answer = gemini_api.call_gemini_with_media(media.name, "describe")

        self.assertEqual(answer, "answer")
        self.assertEqual(post.call_count, 2)

    def test_429_backoff_stops_after_six_attempts_and_caps_at_60_seconds(self):
        error = _FakeErrorResponse()
        error.status_code = 429
        error.text = '{"error":{"message":"rate limit exceeded"}}'
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            media.write(b"test media")
            media.flush()
            with (
                patch.object(gemini_api, "YOUR_API_KEY_GEMINI", "test-key"),
                patch.object(gemini_api, "GEMINI_BASE_URL", "https://gemini.example"),
                patch.object(gemini_api, "GEMINI_MAX_RETRIES", 6),
                patch.object(gemini_api, "GEMINI_RETRY_DELAY_S", 5),
                patch.object(gemini_api, "GEMINI_RETRY_MAX_DELAY_S", 60),
                patch.object(
                    gemini_api.requests, "post", return_value=error
                ) as post,
                patch.object(gemini_api.time, "sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "rate limit exceeded"):
                    gemini_api.call_gemini_with_media(media.name, "describe")

        self.assertEqual(post.call_count, 6)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [5, 10, 20, 40, 60],
        )

    def test_other_400_error_is_not_retried(self):
        error = _FakeErrorResponse()
        error.text = '{"error":{"message":"unsupported MIME type"}}'
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            media.write(b"test media")
            media.flush()
            with (
                patch.object(gemini_api, "YOUR_API_KEY_GEMINI", "test-key"),
                patch.object(gemini_api, "GEMINI_BASE_URL", "https://gemini.example"),
                patch.object(gemini_api, "GEMINI_MAX_RETRIES", 6),
                patch.object(gemini_api.requests, "post", return_value=error) as post,
            ):
                with self.assertRaisesRegex(RuntimeError, "unsupported MIME type"):
                    gemini_api.call_gemini_with_media(media.name, "describe")

        self.assertEqual(post.call_count, 1)

    def test_gemini_usage_and_retry_adjusted_latency_are_recorded(self):
        error = _FakeErrorResponse()
        error.status_code = 429
        error.text = '{"error":{"message":"rate limit exceeded"}}'
        recorder = RolloutTraceRecorder()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as media:
            media.write(b"test media")
            media.flush()
            with (
                patch.object(gemini_api, "YOUR_API_KEY_GEMINI", "test-key"),
                patch.object(gemini_api, "GEMINI_BASE_URL", "https://gemini.example"),
                patch.object(gemini_api, "GEMINI_MAX_RETRIES", 2),
                patch.object(gemini_api, "GEMINI_RETRY_DELAY_S", 0),
                patch.object(
                    gemini_api.requests,
                    "post",
                    side_effect=[error, _FakeResponse()],
                ),
                patch.object(gemini_api.time, "sleep"),
                patch.object(
                    latency.time,
                    "perf_counter",
                    side_effect=[0.0, 2.0, 5.0, 9.0, 10.0],
                ),
                latency.sample_latency() as tracker,
                activate_trace_recorder(recorder),
            ):
                answer = gemini_api.call_gemini_with_media(media.name, "describe")
                summary = tracker.summary(complete=True)

        self.assertEqual(answer, "answer")
        self.assertEqual(
            recorder.perception_calls[0]["usage_metadata"]["totalTokenCount"],
            120,
        )
        self.assertEqual(summary["wall_clock_seconds"], 10.0)
        self.assertEqual(summary["retry_adjusted_seconds"], 5.0)
        self.assertEqual(summary["perception_successful_calls"], 1)

    def test_brain_keeps_model_and_omits_custom_base_url(self):
        calls = []

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = FakeChatOpenAI
        with patch.dict(sys.modules, {"langchain_openai": fake_module}):
            sys.modules.pop("omni_agent.brain", None)
            brain = importlib.import_module("omni_agent.brain")
            with (
                patch.object(brain, "OPENAI_API_KEY", "elm-key"),
                patch.object(brain, "BRAIN_MODEL", "o3"),
            ):
                brain.get_brain_llm()

        self.assertEqual(calls[0]["model"], "o3")
        self.assertEqual(calls[0]["api_key"], "elm-key")
        self.assertEqual(calls[0]["reasoning_effort"], "high")
        self.assertNotIn("base_url", calls[0])
        self.assertNotIn("openai_api_base", calls[0])


if __name__ == "__main__":
    unittest.main()
