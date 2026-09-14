from typing import Dict, Optional, List, Tuple
import os
import base64
import numpy as np
import cv2  # pip install opencv-python
from moviepy import VideoFileClip


def legalize_time_range(t_start, t_end, video_end):
    """Return a valid clip range while preserving fully out-of-bounds duration.

    Ranges that overlap the video are clamped to its boundaries. Ranges wholly
    before or after the video are shifted onto the nearest boundary while
    preserving their original duration, capped by the full video duration.
    """
    original_start = float(t_start)
    original_end = float(t_end)
    video_end = float(video_end)
    original_duration = original_end - original_start

    if video_end <= 0:
        raise ValueError(f"Video duration must be positive, got {video_end}")
    if original_duration <= 0:
        raise ValueError(
            "Clip end time must be greater than start time, got "
            f"({original_start}, {original_end})"
        )

    if original_start >= video_end:
        legal_end = video_end
        legal_start = max(0.0, video_end - original_duration)
    elif original_end <= 0:
        legal_start = 0.0
        legal_end = min(original_duration, video_end)
    else:
        legal_start = max(original_start, 0.0)
        legal_end = min(original_end, video_end)

    if legal_end <= legal_start:
        raise ValueError(
            f"Could not construct a valid clip range from ({original_start}, "
            f"{original_end}) for a {video_end}-second video"
        )
    return legal_start, legal_end


def cut_video(in_path, out_path, t_start, t_end):
    with VideoFileClip(in_path) as video:
        legal_start, legal_end = legalize_time_range(
            t_start, t_end, video.duration
        )
        sub = video.subclipped(legal_start, legal_end)
        sub.write_videofile(
            out_path,
            codec="libx264", 
            audio=False,       
            logger=None       
        )
    return legal_start, legal_end
