# weights/

Model checkpoints go here (git-ignored — download separately).

| file | purpose | source |
|------|---------|--------|
| `yolov8n-face.pt` | YOLOv8-face detector | https://github.com/akanametov/yolov8-face |
| `adaface_ir50_ms1mv2.ckpt` | AdaFace IR-50 ReID (MS1MV2) | AdaFace model zoo — https://github.com/mk-minchul/AdaFace#pretrained-models |

PIPNet (ResNet-18, 300W) landmark weights are fetched automatically by `torchlm`
on first use — nothing to place here for landmarks.

## Overriding paths

```bash
export TRACE_YOLO=/path/to/yolov8n-face.pt
export TRACE_ADAFACE=/path/to/adaface_ir50_ms1mv2.ckpt
```
