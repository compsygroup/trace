"""Export tracks to 3DI/bitbox files: <stem>_id<tid>_rects.3DI and _landmarks.3DI.

Rects are `x y w h`; landmarks are the iBUG-51 subset (pts68[17:68], 102 fields).
Output is dense (one row per video frame); undetected frames are sentinels.
Coordinates are raw-frame pixels (not camera-undistorted).
"""
import json
from pathlib import Path

IBUG51_START, IBUG51_END = 17, 68
N_LM = IBUG51_END - IBUG51_START

_RECT_SENTINEL = "0 0 0 0"
_LM_SENTINEL = " ".join(["nan"] * (N_LM * 2))


def _select_tracks(result, track_id=None, dominant_only=False):
    tracks = result.get("tracks", {})
    if not tracks:
        return []
    if track_id is not None:
        key = str(track_id)
        return [key] if key in tracks else []
    if dominant_only:
        return [max(tracks, key=lambda k: tracks[k].get("median_area", 0))]
    return list(tracks.keys())


def _rect_row(bbox):
    if not bbox or len(bbox) < 4:
        return _RECT_SENTINEL
    x1, y1, x2, y2 = bbox[:4]
    return f"{int(round(x1))} {int(round(y1))} {int(round(x2 - x1))} {int(round(y2 - y1))}"


def _lm_row(lm, prec=3):
    if not lm or len(lm) < IBUG51_END:
        return _LM_SENTINEL
    vals = []
    for x, y in lm[IBUG51_START:IBUG51_END]:
        vals.append(f"{x:.{prec}f}")
        vals.append(f"{y:.{prec}f}")
    return " ".join(vals)


def _rows(track, frame_count, dense, prec):
    bboxes = track.get("bboxes", [])
    landmarks = track.get("landmarks", [])
    frames = track.get("frames", [])

    if dense and frame_count and frames and len(frames) == len(bboxes):
        rects = [_RECT_SENTINEL] * frame_count
        lms = [_LM_SENTINEL] * frame_count
        for i, fi in enumerate(frames):
            if 0 <= fi < frame_count:
                rects[fi] = _rect_row(bboxes[i])
                lms[fi] = _lm_row(landmarks[i] if i < len(landmarks) else None, prec)
        return rects, lms

    rects = [_rect_row(b) for b in bboxes]
    lms = [_lm_row(landmarks[i] if i < len(landmarks) else None, prec)
           for i in range(len(bboxes))]
    return rects, lms


def _sidecar(result, track, kind, out_file, dense, n_rows):
    return {
        "kind": kind,
        "dense": bool(dense),
        "coordinate_space": "raw_video",
        "columns": "x y w h" if kind == "rects" else "51 x,y pairs (ibug51)",
        "n_rows": n_rows,
        "frame_count": result.get("frame_count"),
        "src_fps": result.get("src_fps"),
        "sample_fps": result.get("sample_fps"),
        "detected_frames": track.get("frames", []),
        "source_video": result.get("video"),
        "frame_size": result.get("frame_size"),
        "output": out_file.name,
    }


def export_3di(result, out_dir, stem=None, track_id=None, dominant_only=False,
               dense=True, prec=3):
    """Write rects + landmarks (+ sidecars), one file pair per identity."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or Path(result.get("video", "video")).stem
    frame_count = result.get("frame_count")
    keys = _select_tracks(result, track_id, dominant_only)
    written = []

    for k in keys:
        track = result["tracks"][k]
        rects, lms = _rows(track, frame_count, dense, prec)

        rects_path = out_dir / f"{base}_id{k}_rects.3DI"
        lm_path = out_dir / f"{base}_id{k}_landmarks.3DI"
        rects_path.write_text("\n".join(rects) + "\n")
        lm_path.write_text("\n".join(lms) + "\n")
        used_dense = dense and bool(frame_count) and bool(track.get("frames"))
        (out_dir / f"{rects_path.name}.json").write_text(
            json.dumps(_sidecar(result, track, "rects", rects_path, used_dense, len(rects)), indent=2))
        (out_dir / f"{lm_path.name}.json").write_text(
            json.dumps(_sidecar(result, track, "landmarks", lm_path, used_dense, len(lms)), indent=2))
        written += [rects_path, lm_path]

    return written
