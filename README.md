# facetrack

Face detection, identity tracking, and 68-point landmarks for video. It detects
faces in each frame, assigns a stable track id to each person across the clip, and
outputs per-frame bounding boxes and landmarks. It does not do any quality/FIQA
scoring; the output is boxes and landmarks for a downstream stage to use.

It is the tracking part of a larger FaceQA pipeline, pulled out and made
general (no "pick the mother/parent" logic, no quality models).

## Pipeline

| # | Stage | Model | Notes |
|---|-------|-------|-------|
| 1 | Detection | YOLOv8-face (`yolov8n-face.pt`) | sampled at `sample_fps` (default 5); boxes below `min_box_area` dropped |
| 2 | Tracking | BoT-SORT (Kalman + camera-motion comp.) | persistent integer track ids |
| 3 | Re-ID | AdaFace IR-50 (512-D) | appearance matching inside the tracker |
| 4 | Landmarks | PIPNet ResNet-18 (68-pt, 300W) | on the tracked crop, stored in full-frame coords |
| 5 | Track merge | AdaFace + temporal + geometric gates | joins one identity split across gaps |
| 6 | Cleanup | — | drops tracks shorter than `min_frames` |

## Install

```bash
pip install -r requirements.txt          # install torch/torchvision for your CUDA first
./setup.sh                               # fetch BoT-SORT + AdaFace into third_party/
# download weights into weights/  (see weights/README.md)
```

## Usage

Command line:

```bash
python -m facetrack.cli clip.mp4 -o out/
python -m facetrack.cli videos/*.mp4 -o out/ --sample-fps 5 --min-box-area 1500
python -m facetrack.cli clip.mp4 -o out/ --no-3di            # tracks JSON only
```

Per video it writes `out/<stem>_tracks.json` plus the 3DI/bitbox files
(`_rects.3DI` / `_landmarks.3DI`) for each identity. Use `--no-3di` to skip them.

Python:

```python
from facetrack import run_tracking, save_result

result = run_tracking("clip.mp4", sample_fps=5, min_box_area=1500)
save_result(result, "clip_tracks.json")
```

## Output schema

```jsonc
{
  "video": "clip.mp4",
  "duration": 55.0,              // seconds
  "frame_size": [1920, 1080],
  "sample_fps": 5.0,            // effective sampled rate
  "n_tracks": 2,
  "tracks": {
    "<track_id>": {
      "n_frames": 240,
      "median_area": 41000,
      "best_crop_b64": "data:image/jpeg;base64,...",  // largest crop thumbnail
      "times":     [ 0.0, 0.2, ... ],                 // seconds, per sampled frame
      "det_conf":  [ 0.94, ... ],                      // YOLO confidence
      "bboxes":    [ [x1, y1, x2, y2, fw, fh], ... ],  // absolute px + frame size
      "landmarks": [ [[x, y], ...68], ... ],           // full-frame coords, or null
      "merged_from":   [3, 7],                         // original ids fused into this track
      "sub_track_ids": [ "3", "3", "7", ... ]          // origin id per frame
    }
  }
}
```

Arrays are frame-aligned: index `i` is the `i`-th sampled frame of that track.

## Parameters

| arg | default | meaning |
|-----|---------|---------|
| `--sample-fps` | 5.0 | frames analysed per second |
| `--min-box-area` | 1500 | discard detections smaller than this (px²) |
| `--merge-threshold` | 0.40 | cosine similarity to merge two tracks into one identity |
| `--min-frames` | 5 | drop tracks with fewer sampled frames |
| `--no-thumbnails` | off | skip embedding track thumbnails |

## Layout

```
facetrack/
├── facetrack/
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

## 3DI / bitbox export

By default the CLI writes one file pair per identity (`<tid>` is the track id);
`--no-3di` disables it, and `facetrack.export_3di.export_3di(result, out_dir)`
does the same from Python:

| file | content |
|------|---------|
| `<stem>_id<tid>_rects.3DI` | `x y w h` (top-left + size, integer px) |
| `<stem>_id<tid>_landmarks.3DI` | 51 landmarks = 102 fields, iBUG-68 inner face `pts68[17:68]` (no jaw, no z) |
| `*.3DI.json` | sidecars with frame indices and coordinate space |

These load with the bitbox readers
([compsygroup/bitbox](https://github.com/compsygroup/bitbox)): `read_rectangles`
returns `x y w h`, `read_landmarks` returns schema `ibug51`.

Output is dense by default: one row per video frame `[0, frame_count)`, indexed by
frame number. Frames with no detection for that identity are written as sentinels
(`0 0 0 0` and `nan`). Since the pipeline runs on sampled frames, most rows are
sentinels unless you process every frame (set `--sample-fps` to the video fps).
`--3di-sparse` writes one row per detection instead.

By default every identity is exported; `--3di-track <id>` limits to one, and
`--3di-dominant` to the largest track.

Coordinates are raw-frame pixels. 3DI-lite runs on the undistorted
`_preprocessed.mp4`, so the files are format-compatible but not pixel-registered to
that output unless you undistort with the same camera model first.

## Notes

- Landmarks in the JSON are iBUG-68 in full-frame coordinates; the `.3DI` export
  keeps the 51-point inner-face subset.
- Quality/FIQA scoring is not part of this repo.
