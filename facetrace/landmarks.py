"""PIPNet 68-point landmarks and ArcFace 5-point alignment."""
import base64

import cv2
import numpy as np
import torch
from skimage import transform as sk_trans

from facetrace.config import (
    ARCFACE_REF, PIP_TF,
    LM68_LEFT_EYE, LM68_RIGHT_EYE, LM68_NOSE_TIP, LM68_MOUTH_L, LM68_MOUTH_R,
)


def pipnet_lm68(model, bgr, device):
    """68x2 landmarks in crop pixel coordinates."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    inp = PIP_TF(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_map, ox, oy = model(inp)[:3]
    B, C, H, W = cls_map.shape
    flat = cls_map.view(B, C, -1)
    idx = torch.argmax(flat, 2)
    ix, iy = idx % W, idx // W
    dx = torch.gather(ox.view(B, C, -1), 2, idx.unsqueeze(-1)).squeeze(-1)
    dy = torch.gather(oy.view(B, C, -1), 2, idx.unsqueeze(-1)).squeeze(-1)
    lx = ((ix + dx) * 32.0).cpu().numpy()[0] * (w / 256.0)
    ly = ((iy + dy) * 32.0).cpu().numpy()[0] * (h / 256.0)
    return np.stack([lx, ly], axis=-1)


def lm68_to_5pt(lm):
    return np.array([
        lm[LM68_LEFT_EYE].mean(0), lm[LM68_RIGHT_EYE].mean(0),
        lm[LM68_NOSE_TIP], lm[LM68_MOUTH_L], lm[LM68_MOUTH_R],
    ], dtype=np.float32)


def align_arcface(bgr, lm5):
    t = sk_trans.SimilarityTransform()
    if not t.estimate(lm5, ARCFACE_REF):
        return None
    return cv2.warpAffine(bgr, t.params[:2].astype(np.float32), (112, 112), borderValue=0)


def img_b64(bgr, quality=88):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
