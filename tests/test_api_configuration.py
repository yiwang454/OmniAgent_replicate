from __future__ import annotations

import base64
import importlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from omni_agent import config
from omni_agent import gemini_api


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
            ]
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
