# third_party/

Two trackers are used as source dependencies (not pip packages). Place them
here so `facetrace/models.py` can add them to `sys.path`:

```
third_party/
├── BoT-SORT/     # provides  tracker.mc_bot_sort.BoTSORT
└── AdaFace/      # provides  net.build_model("ir_50")
```

## Get them

```bash
./setup.sh                       # clones both from upstream
# or symlink existing local checkouts:
TRACE_LOCAL_SRC=/path/to/repo ./setup.sh
```

Upstream:
- BoT-SORT — https://github.com/NirAharon/BoT-SORT
- AdaFace  — https://github.com/mk-minchul/AdaFace

## Overriding the location

If the checkouts live elsewhere, point to them instead of copying:

```bash
export TRACE_BOTSORT_ROOT=/path/to/BoT-SORT
export TRACE_ADAFACE_ROOT=/path/to/AdaFace
```

These directories are git-ignored — they are not committed to this repo.
