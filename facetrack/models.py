"""Load and cache the models: YOLOv8-face, BoT-SORT + AdaFace, PIPNet."""
import sys
import threading

import cv2
import numpy as np
import torch

import facetrack.compat  # noqa: F401  — before the vendored imports

from facetrack.config import (
    ADAFACE_CKPT, YOLO_PT, BOTSORT_ROOT, ADAFACE_ROOT, botsort_args,
)

for _p in (str(BOTSORT_ROOT), str(ADAFACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultralytics import YOLO                     # noqa: E402
from tracker.mc_bot_sort import BoTSORT          # noqa: E402
from torchlm.models import pipnet                # noqa: E402
import net as adaface_net                        # noqa: E402

_models: dict = {}
_models_lock = threading.Lock()


class AdaFaceWrapper:
    """ReID encoder for BoT-SORT: face crop -> 512-D AdaFace embedding."""

    def __init__(self, enc, device):
        self.encoder, self.device = enc, device

    def inference(self, img, bboxes):
        feats = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                t = torch.tensor(((cv2.resize(crop, (112, 112)) / 255.) - 0.5) / 0.5
                                 ).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    f, _ = self.encoder(t)
                feats.append(f[0].cpu().numpy())
            else:
                feats.append(np.zeros(512))
        return np.array(feats)


def load_models(device):
    detector = YOLO(str(YOLO_PT))

    raw_enc = adaface_net.build_model("ir_50")
    sd = torch.load(str(ADAFACE_CKPT), map_location=device, weights_only=False)["state_dict"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    raw_enc.load_state_dict(sd)
    raw_enc = raw_enc.eval().to(device)
    tracker_encoder = AdaFaceWrapper(raw_enc, device)

    landmarker = pipnet(backbone="resnet18", pretrained=True,
                        num_lms=68, meanface_type="300w").to(device).eval()

    return dict(
        detector=detector,
        ada_enc=raw_enc,
        tracker_encoder=tracker_encoder,
        landmarker=landmarker,
        device=device,
    )


def build_tracker(models, min_box_area, device):
    """Fresh BoT-SORT per video with the shared AdaFace ReID encoder."""
    tracker = BoTSORT(botsort_args(min_box_area, device), frame_rate=30)
    tracker.encoder = models["tracker_encoder"]
    tracker.args.with_reid = True
    return tracker


def get_models(device=None):
    with _models_lock:
        if not _models:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _models.update(load_models(device))
        return _models
