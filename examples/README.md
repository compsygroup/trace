# examples/

Run outputs land here by default:

```bash
python -m facetrace.cli your_clip.mp4 -o examples/
```

`*_tracks.json` files are git-ignored — they can contain face-crop thumbnails
(PII), so they are never committed. See the output schema in the top-level
[README](../README.md#output-schema).
