#!/usr/bin/env bash
set -euo pipefail
BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"; cd "$DEST"
for f in \
  body-annotations-male-cns-v1.0-minconf-0.5.feather \
  body-neurotransmitters-male-cns-v1.0.feather \
  connectome-weights-male-cns-v1.0-minconf-0.5.feather ; do
  echo ">> $f"; curl -fL -O --retry 3 -C - "$BASE/$f"
done
