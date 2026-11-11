# TRACE

**T**rack, **R**e-identify, **A**lign, **C**lean, **E**xport.

Face detection, identity tracking, and 51-point (bitbox ibug51) landmarks for video. It detects
faces in each frame, assigns a stable track id to each person across the clip, and
outputs per-frame bounding boxes and landmarks.

## Pipeline

| # | Stage | Model | Notes |
|---|-------|-------|-------|
| 1 | Detection | YOLOv8-face (`yolov8n-face.pt`) | every frame; boxes below `min_box_area` dropped |
| 2 | Tracking | BoT-SORT (Kalman + camera-motion comp.) | persistent integer track ids |
| 3 | Re-ID | AdaFace IR-50 (512-D) | appearance matching inside the tracker |
| 4 | Landmarks | PIPNet ResNet-18 (68-pt, 300W) | on the tracked crop; saved as ibug51 (jaw dropped) in full-frame coords |
| 5 | Track merge | AdaFace + temporal + geometric gates | joins one identity split across gaps |
| 6 | Cleanup | — | drops tracks detected in fewer than `min_track_frames` frames |

## Install

```bash
pip install -r requirements.txt          # install torch/torchvision for your CUDA first
./setup.sh                               # fetch BoT-SORT + AdaFace into third_party/
# download weights into weights/  (see weights/README.md)
```

## Usage

Command line:

```bash
python -m facetrace.cli clip.mp4 -o out/
python -m facetrace.cli videos/*.mp4 -o out/ --min-box-area 1500
python -m facetrace.cli session.mp4 -o out/ --start 5:00 --end 10:00   # segment only (also 300, or 0:05:00)
```

Per video it writes one file, `out/<stem>_tracks.json`.

Python:

```python
from facetrace import run_tracking, save_result

result = run_tracking("clip.mp4", min_box_area=1500)
save_result(result, "clip_tracks.json")
```

## Output schema

```jsonc
{
  "video": "clip.mp4",
  "duration": 55.0,              // seconds
  "frame_size": [1920, 1080],
  "src_fps": 30.0,              // native fps from container metadata
  "frame_count": 9000,          // frames processed
  "start_frame": 9000,          // segment start in the full video (round(start * src_fps))
  "end_frame": 18000,           // segment end (exclusive); frame i here = full-video frame start_frame + i
  "n_tracks": 2,
  "tracks": {
    "<track_id>": {
      "n_frames": 240,
      "median_area": 41000,
      "best_crop_b64": "data:image/jpeg;base64,...",  // only with --save-thumbnail: largest face crop
      "det_conf":  [ 0.94, ... ],                      // YOLO confidence, 0.0 if not detected
      "bboxes":    [ [x, y, w, h], ... ],              // top-left + size, px; [0,0,0,0] if not detected
      "landmarks": [ [[x, y], ...51], ... ],           // ibug51, full-frame px; all zeros if not detected
      "merged_from":   [3, 7],                         // original ids fused into this track
      "sub_track_ids": [ "3", "3", "7", null, ... ]    // origin id per frame, null if not detected
    }
  }
}
```

Per-track arrays are dense: length = `frame_count`, index `i` = processed frame `i`
(full-video frame `start_frame + i`). Undetected frames are zero-filled.
`n_frames` counts detected (non-zero) frames.

## Parameters

| arg | default | meaning |
|-----|---------|---------|
| `--start` | start of video | segment start: seconds, `MM:SS` or `HH:MM:SS` |
| `--end` | end of video | segment end (exclusive), same formats |
| `--min-box-area` | 1500 | discard detections smaller than this (px²) |
| `--merge-threshold` | 0.40 | cosine similarity to merge two tracks into one identity |
| `--min-track-frames` | 5 | drop tracks detected in fewer frames than this |
| `--save-thumbnail` | off | embed each track's largest face crop (160x160 base64 JPEG) |

## Layout

```
trace/
├── facetrace/
│   ├── config.py        # paths, tracker params, ArcFace/landmark constants
│   ├── models.py        # load YOLO + BoT-SORT/AdaFace + PIPNet
│   ├── landmarks.py     # PIPNet 68-pt + ArcFace alignment
│   ├── track_merger.py  # same-identity track merging
│   ├── pipeline.py      # run_tracking()
│   └── cli.py           # command-line entry point
├── third_party/         # BoT-SORT + AdaFace (fetched by setup.sh)
├── weights/             # model checkpoints (downloaded separately)
└── requirements.txt
```
