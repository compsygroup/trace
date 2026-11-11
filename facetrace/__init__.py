"""facetrace — face detection + identity tracking + 51-point (ibug51) landmarks.

Import the high-level entry point directly:

    from facetrace import run_tracking, save_result
    result = run_tracking("clip.mp4")
    save_result(result, "clip_tracks.json")
"""
from facetrace.pipeline import run_tracking, save_result  # noqa: F401

__version__ = "0.1.0"
__all__ = ["run_tracking", "save_result"]
