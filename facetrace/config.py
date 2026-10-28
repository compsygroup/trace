"""Paths, tracker parameters and landmark/alignment constants.

Weight and third-party paths default to the repo layout; override with env vars
TRACE_YOLO, TRACE_ADAFACE, TRACE_BOTSORT_ROOT, TRACE_ADAFACE_ROOT.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torchvision.transforms as T

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
WEIGHTS_DIR = Path(os.environ.get("TRACE_WEIGHTS", REPO_ROOT / "weights"))
THIRD_PARTY = REPO_ROOT / "third_party"

YOLO_PT = Path(os.environ.get("TRACE_YOLO", WEIGHTS_DIR / "yolov8n-face.pt"))
ADAFACE_CKPT = Path(os.environ.get("TRACE_ADAFACE", WEIGHTS_DIR / "adaface_ir50_ms1mv2.ckpt"))

BOTSORT_ROOT = Path(os.environ.get("TRACE_BOTSORT_ROOT", THIRD_PARTY / "BoT-SORT"))
ADAFACE_ROOT = Path(os.environ.get("TRACE_ADAFACE_ROOT", THIRD_PARTY / "AdaFace"))

DEFAULT_SAMPLE_FPS = 5.0
DEFAULT_MIN_BOX_AREA = 1500
DEFAULT_MERGE_THRESHOLD = 0.40
DEFAULT_MIN_FRAMES = 5


def botsort_args(min_box_area=DEFAULT_MIN_BOX_AREA, device="cpu"):
    return SimpleNamespace(
        name="facetrace", ablation=False,
        track_high_thresh=0.5, track_low_thresh=0.1,
        new_track_thresh=0.6, match_thresh=0.8,
        track_buffer=30, aspect_ratio_thresh=1.6,
        min_box_area=min_box_area, mot20=False,
        cmc_method="sparseOptFlow",
        with_reid=False,
        proximity_thresh=0.5, appearance_thresh=0.25,
        fast_reid_config=None, fast_reid_weights=None,
        device=device,
    )


# ArcFace 5-point reference (112x112) for aligning crops before the ReID embedding
ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)

LM68_LEFT_EYE = list(range(36, 42))
LM68_RIGHT_EYE = list(range(42, 48))
LM68_NOSE_TIP = 30
LM68_MOUTH_L = 48
LM68_MOUTH_R = 54

PIP_TF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
