from langchain_core.tools import ToolException
from .units import cut_video
from langchain_core.tools import tool
from typing import Dict
from .video_llm import video_llm_gemini
from omni_agent.tool_outcomes import invalid_tool_input

@tool
def video_global_qa(
    video_path: str,
    question: str,
) -> Dict[str, str]:
    """
    Answer a free-form question about the overall visual content of a video.

    The agent can call this tool multiple times with different questions.
    This tool:
    1) Uniformly samples a set of frames from the entire duration of the video.
    2) Uses VideoLLM to inspect those frames and
       answer the question about what happens in the video as a whole.

    Args:
        video_path: Path or identifier of the video file.
        question: Any natural-language question about the global visual content.

    Returns:
        A dictionary with:
            - "answer": the model's answer to the question, based on the sampled frames.
    """

    text_block = (
        "You will be shown a video.\n\n"
        + "Your tasks are:\n"
          "1. Carefully inspect what is happening in the video.\n"
          "2. Reason step by step if necessary.\n"
          "3. Answer the user's question as precisely as possible, "
          "always staying consistent with what is visible in the video.\n\n"
        f"User question:\n{question}\n\n"
    )

    try:
        visual_notes = video_llm_gemini(video_path, text_block)
    except Exception as e:
        raise ToolException(f"[video_global_qa error] {type(e).__name__}: {e}")
    return {
        "answer": visual_notes,
    }


@tool
def video_clip_qa(
    video_path: str,
    question: str,
    time_range,
) -> Dict[str, object]:
    """
    Answer a free-form question about the visual content of a specific time range
    (clip) within a video.

    The agent can call this tool multiple times with different sub-questions.
    This tool:
    1) Densely samples a set of frames from the specified time range of the video.
    2) Uses VideoLLM to inspect those frames and
       answer the question about what happens in that clip.

    Args:
        video_path: Path or identifier of the video file.
        question: Any natural-language question about the visual content of the clip.
        time_range: A tuple (start_time, end_time) in seconds with start_time < end_time.
            Values outside the video are automatically clamped. A range wholly
            before or after the video is shifted to the nearest boundary while
            preserving its original duration when the video is long enough.
            - time_range is mandatory for this tool: it MUST NOT be None.

    Returns:
        A dictionary with:
            - "answer": the model's answer to the question, focused on that time range.
    """
    try:
        original_start = float(time_range[0])
        original_end = float(time_range[1])
    except (TypeError, ValueError, IndexError) as exc:
        return invalid_tool_input(
            "Invalid clip time_range: expected two numeric values "
            f"(start, end). Details: {exc}"
        )

    if original_end <= original_start:
        return invalid_tool_input(
            "Invalid clip duration: end <= start "
            f"({original_end:g} <= {original_start:g}). Choose a new time_range "
            "whose end is greater than its start."
        )

    import random
    import os
    cache_path = f"Cache/clip_video_{random.randint(0, 1_000_000_000)}.mp4"
    try:
        start, end = cut_video(
            video_path, cache_path, original_start, original_end
        )
        clip_context = (
            "You are analyzing a short VIDEO CLIP taken from a longer video.\n"
            f"It corresponds to the time range roughly from {start:.2f} to {end:.2f} "
            "seconds in the original video.\n"
            "The frames are in temporal order and show how the scene evolves across "
            "this clip.\n"
            "Assume the frames have ALREADY been correctly aligned with this time "
            "range; do not claim that they are from an earlier or later part of the "
            "video.\n"
            "Answer ONLY about what happens within this clip. If the requested event "
            "is not visible here, say that it is not visible in this clip.\n\n"
        )

        text_block = (
            "You will be shown a video.\n"
            "Treat them as a short video: reason about how objects and people change "
            "over time across the frames, not just each image in isolation.\n\n"
            "Your tasks:\n"
            "1. Understand the main actions and changes that occur during this clip.\n"
            "2. Reason step by step if needed.\n"
            "3. Answer the user's question as precisely as possible, always staying "
            "consistent with what is visually supported in the frames.\n\n"
            + clip_context
            + f"User question:\n{question}\n\n"
        )
        visual_notes = video_llm_gemini(cache_path, text_block, fps=5)
    except Exception as e:
        raise ToolException(f"[video_clip_qa error] {type(e).__name__}: {e}")
    finally:
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
    return {
        "answer": visual_notes,
    }

if __name__ == "__main__":
    out = video_clip_qa.invoke({"video_path": "assert/026dzf-vc5g_video.mp4", "question": "Describe the video in detail.", "time_range": (45,46)})
    print(out)
