"""Detection -> tracking (ReID) -> landmarks -> identity merge, per video."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from facetrace.config import (
    DEFAULT_MIN_BOX_AREA, DEFAULT_MERGE_THRESHOLD, DEFAULT_MIN_FRAMES,
)
from facetrace.landmarks import pipnet_lm68, lm68_to_5pt, align_arcface, img_b64
from facetrace.models import get_models, build_tracker
from facetrace.track_merger import merge_same_person_tracks


def _empty_result(video_path, duration, width, height, src_fps, frame_count, start_frame):
    return {
        "video": Path(video_path).name,
        "duration": duration,
        "frame_size": [width, height],
        "src_fps": src_fps,
        "frame_count": frame_count,
        "start_frame": start_frame,
        "end_frame": start_frame + frame_count,
        "n_tracks": 0,
        "tracks": {},
    }


def run_tracking(video_path, start_frame=0, end_frame=None,
                 min_box_area=DEFAULT_MIN_BOX_AREA,
                 merge_threshold=DEFAULT_MERGE_THRESHOLD,
                 min_frames=DEFAULT_MIN_FRAMES,
                 save_thumbnails=True, models=None, device=None, progress=None):
    """Process one video, or frames [start_frame, end_frame) of it, and return the
    per-track detection/landmark result dict. Frame indices in the result are
    relative to start_frame."""
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError(f"invalid segment: start_frame={start_frame}, end_frame={end_frame}")
    video_path = Path(video_path)

    def _prog(msg):
        if progress:
            progress(msg)

    _prog("Loading models…")
    m = models or get_models(device)
    detector, landmarker = m["detector"], m["landmarker"]
    raw_enc, device = m["ada_enc"], m["device"]

    tracker = build_tracker(m, min_box_area, device)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # skip to start_frame by decoding (grab), not seeking: CAP_PROP_POS_FRAMES
    # seeks are not frame-accurate on all codecs
    for _ in range(start_frame):
        if not cap.grab():
            break
    n_seg = (min(end_frame, n_frames) if end_frame else n_frames) - start_frame

    track_data = defaultdict(lambda: dict(
        frames=[], times=[], areas=[], det_conf=[], bboxes=[], landmarks=[],
        best_crop_b64="", best_area=0, top_embs=[], best_emb=None))

    frame_idx, seen = 0, 0
    while cap.isOpened():
        if end_frame is not None and start_frame + frame_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector(frame, verbose=False)[0].boxes.data.cpu().numpy()
        if len(dets) > 0:
            bboxes, scores_ = dets[:, :4], dets[:, 4]
            classes = np.zeros_like(scores_)
            tracks = tracker.update(np.column_stack((bboxes, scores_, classes)), frame)
            for trk in tracks:
                x1, y1, x2, y2 = [max(0, int(v)) for v in trk.tlbr]
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                area = (x2 - x1) * (y2 - y1)
                if area < min_box_area:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                tid = int(trk.track_id)
                t_sec = frame_idx / src_fps
                conf = float(getattr(trk, "score", 0.0) or 0.0)

                lm68_frame, aligned = None, None
                try:
                    lm68 = pipnet_lm68(landmarker, crop, device)
                    aligned = align_arcface(crop, lm68_to_5pt(lm68))
                    if aligned is None:
                        aligned = cv2.resize(crop, (112, 112))
                    lm68_frame = (lm68 + np.array([x1, y1], dtype=np.float32)).tolist()
                except Exception:
                    aligned = cv2.resize(crop, (112, 112))

                td = track_data[tid]
                td["frames"].append(frame_idx)
                td["times"].append(t_sec)
                td["areas"].append(area)
                td["det_conf"].append(conf)
                td["bboxes"].append([x1, y1, x2, y2, width, height])
                td["landmarks"].append(lm68_frame)
                if save_thumbnails and area > td["best_area"]:
                    td["best_area"] = area
                    td["best_crop_b64"] = img_b64(cv2.resize(crop, (160, 160)))

                # keep the top-K embeddings, ranked by face area
                top, K = td["top_embs"], 5
                min_kept = top[-1][0] if len(top) >= K else -1.0
                if area > min_kept:
                    with torch.no_grad():
                        t_in = torch.tensor(
                            ((aligned.astype(np.float32) / 255.) - 0.5) / 0.5
                        ).permute(2, 0, 1).unsqueeze(0).float().to(device)
                        emb, _ = raw_enc(t_in)
                    top.append((float(area), emb[0].cpu().numpy().tolist()))
                    top.sort(key=lambda x: x[0], reverse=True)
                    if len(top) > K:
                        top.pop()
                    td["best_emb"] = top[0][1]
                seen += 1

        frame_idx += 1
        if frame_idx % 300 == 0:
            pct = round(100 * frame_idx / max(n_seg, 1))
            _prog(f"Frame {frame_idx}/{n_seg if n_seg > 0 else '?'} ({pct}%) — {seen} faces tracked")

    cap.release()
    frame_count = frame_idx
    duration = max(frame_count / src_fps, 1.0)

    if not track_data:
        return _empty_result(video_path, duration, width, height,
                             src_fps, frame_count, start_frame)

    _prog("Merging split tracks…")
    track_data = merge_same_person_tracks(dict(track_data), threshold=merge_threshold)
    track_data = {t: td for t, td in track_data.items()
                  if len(td.get("times", [])) >= min_frames}

    if not track_data:
        return _empty_result(video_path, duration, width, height,
                             src_fps, frame_count, start_frame)

    tracks = {}
    for tid, td in track_data.items():
        # dense, zero-filled: element i = processed frame i
        bboxes = [[0, 0, 0, 0] for _ in range(frame_count)]
        landmarks = [[[0.0, 0.0] for _ in range(68)] for _ in range(frame_count)]
        det_conf = [0.0] * frame_count
        sub_ids = [None] * frame_count
        for f, bb, lm, c, s in zip(td["frames"], td["bboxes"], td["landmarks"],
                                   td["det_conf"], td["sub_track_ids"]):
            # merged sub-tracks may overlap by up to 0.3 s: keep the higher-confidence one
            if any(bboxes[f]) and det_conf[f] >= c:
                continue
            bboxes[f] = bb[:4]
            if lm is not None:
                landmarks[f] = lm
            det_conf[f], sub_ids[f] = c, s
        tracks[str(tid)] = {
            "n_frames": sum(1 for b in bboxes if any(b)),
            "median_area": td.get("median_area",
                                  int(statistics.median(td["areas"])) if td.get("areas") else 0),
            "best_crop_b64": td.get("best_crop_b64", ""),
            "det_conf": det_conf,
            "bboxes": bboxes,
            "landmarks": landmarks,
            "merged_from": td.get("merged_from", [tid]),
            "sub_track_ids": sub_ids,
        }

    return {
        "video": video_path.name,
        "duration": duration,
        "frame_size": [width, height],
        "src_fps": src_fps,
        "frame_count": frame_count,
        "start_frame": start_frame,
        "end_frame": start_frame + frame_count,
        "n_tracks": len(tracks),
        "tracks": tracks,
    }


def _sanitize(o):
    if isinstance(o, float):
        return None if o != o else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def save_result(result, out_path):
    n = result["frame_count"]
    for tid, t in result["tracks"].items():
        assert len(t["bboxes"]) == len(t["landmarks"]) == n, \
            f"track {tid}: {len(t['bboxes'])} bboxes, {len(t['landmarks'])} landmarks, {n} frames"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(result)))
    return out_path
