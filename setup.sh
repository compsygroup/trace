#!/usr/bin/env bash
# Fetch the two vendored third-party trackers into third_party/.
# Weights are downloaded separately — see weights/README.md.
set -euo pipefail
cd "$(dirname "$0")"

TP="third_party"
mkdir -p "$TP"

clone_or_link () {
  name="$1"; url="$2"
  if [ -d "$TP/$name" ]; then
    echo "[skip] $TP/$name already present"
  elif [ -n "${TRACE_LOCAL_SRC:-}" ] && [ -d "$TRACE_LOCAL_SRC/$name" ]; then
    ln -s "$TRACE_LOCAL_SRC/$name" "$TP/$name"
    echo "[link] $TP/$name -> $TRACE_LOCAL_SRC/$name"
  else
    git clone --depth 1 "$url" "$TP/$name"
    echo "[clone] $TP/$name"
  fi
}

# Set TRACE_LOCAL_SRC=/path/to/existing/repo to symlink local copies instead of cloning.
clone_or_link "BoT-SORT" "https://github.com/NirAharon/BoT-SORT.git"
clone_or_link "AdaFace"  "https://github.com/mk-minchul/AdaFace.git"

echo
echo "Third-party trackers ready under $TP/."
echo "Next: pip install -r requirements.txt, then download weights (weights/README.md)."
