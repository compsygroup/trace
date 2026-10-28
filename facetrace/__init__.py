"""facetrace — face detection + identity tracking + 68-point landmarks.

A general-purpose tracking front-end (no quality/FIQA scoring). Import the
high-level entry point directly:

    from facetrace import run_tracking, save_result
    result = run_tracking("clip.mp4")
    save_result(result, "clip_tracks.json")
"""
from facetrace.pipeline import run_tracking, save_result  # noqa: F401
from facetrace.export_3di import export_3di  # noqa: F401

__version__ = "0.1.0"
__all__ = ["run_tracking", "save_result", "export_3di"]
