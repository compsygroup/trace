"""CLI: python -m facetrace.cli VIDEO [VIDEO ...] -o OUT_DIR"""
import argparse
import sys
from pathlib import Path

from facetrace.config import (
    DEFAULT_MIN_BOX_AREA, DEFAULT_MERGE_THRESHOLD, DEFAULT_MIN_FRAMES,
)
from facetrace.pipeline import run_tracking, save_result
from facetrace.models import get_models


def _time(s):
    """Seconds, MM:SS or HH:MM:SS (fractions allowed) -> seconds."""
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"bad time {s!r}: use seconds, MM:SS or HH:MM:SS")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="facetrace",
        description="Face detection + identity tracking + 51-point (ibug51) landmarks.")
    ap.add_argument("videos", nargs="+", type=Path, help="input video file(s)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"),
                    help="output directory for <stem>_tracks.json (default: ./out)")
    ap.add_argument("--start", type=_time, default=None,
                    help="segment start: seconds, MM:SS or HH:MM:SS (default: start of video)")
    ap.add_argument("--end", type=_time, default=None,
                    help="segment end, exclusive (default: end of video)")
    ap.add_argument("--min-box-area", type=int, default=DEFAULT_MIN_BOX_AREA)
    ap.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    ap.add_argument("--no-thumbnails", action="store_true",
                    help="do not embed track thumbnail JPEGs in the output")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    models = get_models()  # load once, reuse across videos

    rc = 0
    for video in args.videos:
        if not video.exists():
            print(f"[skip] not found: {video}", file=sys.stderr)
            rc = 1
            continue
        print(f"[facetrace] {video.name}", flush=True)
        try:
            result = run_tracking(
                video, start=args.start, end=args.end,
                min_box_area=args.min_box_area,
                merge_threshold=args.merge_threshold, min_frames=args.min_frames,
                save_thumbnails=not args.no_thumbnails, models=models,
                progress=lambda msg: print(f"  {msg}", flush=True))
            out_path = save_result(result, args.out / f"{video.stem}_tracks.json")
            print(f"  -> {out_path}  ({result['n_tracks']} tracks)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {video.name}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
