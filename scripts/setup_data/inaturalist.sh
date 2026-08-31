#!/bin/bash
# iNaturalist 2021 (10,000 species) -- fine-grained visual categorization.
# By default the "mini" training set (500K images, 50 per species) is downloaded,
# which is what configs/classification/probing_inat21.yaml expects; set
# SPLIT=train for the full 2.7M-image training set.
set -euo pipefail

DATA_ROOT=${1:-./data}
SPLIT=${SPLIT:-train_mini}
DEST="$DATA_ROOT/inaturalist21"
BASE=https://ml-inat-competition-datasets.s3.amazonaws.com/2021
mkdir -p "$DEST"
cd "$DEST"

download () {  # $1 = archive prefix, $2 = torchvision directory name
    if [ ! -d "$2" ]; then
        wget -c "$BASE/$1.tar.gz"
        wget -c "$BASE/$1.json.tar.gz"
        tar -xzf "$1.tar.gz"
        tar -xzf "$1.json.tar.gz"
        mv "$1" "$2"
        mv "$1.json" "$2.json" 2>/dev/null || true
        rm -f "$1.tar.gz" "$1.json.tar.gz"
    fi
}

download "$SPLIT" "2021_${SPLIT}"
download val 2021_valid

# Expected layout (torchvision.datasets.INaturalist):
#   data/inaturalist21/2021_train_mini/<category>/*.jpg
#   data/inaturalist21/2021_valid/<category>/*.jpg
ls -d 2021_* 
