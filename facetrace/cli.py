"""CLI: python -m facetrace.cli VIDEO [VIDEO ...] -o OUT_DIR"""
import argparse
import sys
from pathlib import Path

from facetrace.config import (
    DEFAULT_SAMPLE_FPS, DEFAULT_MIN_BOX_AREA, DEFAULT_MERGE_THRESHOLD, DEFAULT_MIN_FRAMES,
)
from facetrace.pipeline import run_tracking, save_result
from facetrace.models import get_models
from facetrace.export_3di import export_3di


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="facetrace",
        description="Face detection + identity tracking + 68-point landmarks.")
    ap.add_argument("videos", nargs="+", type=Path, help="input video file(s)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"),
                    help="output directory for <stem>_tracks.json (default: ./out)")
    ap.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    ap.add_argument("--min-box-area", type=int, default=DEFAULT_MIN_BOX_AREA)
    ap.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    ap.add_argument("--no-thumbnails", action="store_true",
                    help="do not embed track thumbnail JPEGs in the output")
    ap.add_argument("--no-3di", dest="no_3di", action="store_true",
                    help="skip the 3DI/bitbox rects + landmarks files (written by default)")
    ap.add_argument("--3di-track", dest="di_track", default=None,
                    help="export only this identity (default: every identity)")
    ap.add_argument("--3di-dominant", dest="di_dominant", action="store_true",
                    help="export only the dominant (largest) identity")
    ap.add_argument("--3di-sparse", dest="di_sparse", action="store_true",
                    help="3DI: one row per sampled detection instead of dense per-frame")
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
                video, sample_fps=args.sample_fps, min_box_area=args.min_box_area,
                merge_threshold=args.merge_threshold, min_frames=args.min_frames,
                save_thumbnails=not args.no_thumbnails, models=models,
                progress=lambda msg: print(f"  {msg}", flush=True))
            out_path = save_result(result, args.out / f"{video.stem}_tracks.json")
            print(f"  -> {out_path}  ({result['n_tracks']} tracks)", flush=True)
            if not args.no_3di:
                files = export_3di(result, args.out, stem=video.stem,
                                   track_id=args.di_track, dominant_only=args.di_dominant,
                                   dense=not args.di_sparse)
                for f in files:
                    print(f"  -> {f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {video.name}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
