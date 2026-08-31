#!/bin/bash
# CLEVR v1.0 -- used for the low-level reasoning transfer (Clevr/Count, Clevr/Dist).
# The VTAB-1k splits (800 train / 200 val / 15,000 test) are derived on the fly
# from the scene graphs by src/dataset/clevr.py, so only the raw release is needed.
set -euo pipefail

DATA_ROOT=${1:-./data}
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

if [ ! -d CLEVR_v1.0 ]; then
    wget -c https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
    unzip -q CLEVR_v1.0.zip
    rm CLEVR_v1.0.zip
fi

# Expected layout:
#   data/CLEVR_v1.0/images/{train,val}/*.png
#   data/CLEVR_v1.0/scenes/CLEVR_{train,val}_scenes.json
ls CLEVR_v1.0/images CLEVR_v1.0/scenes
