#!/bin/bash
# ADE20K (SceneParse150) -- semantic segmentation transfer.
set -euo pipefail

DATA_ROOT=${1:-./data}
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

if [ ! -d ADEChallengeData2016 ]; then
    wget -c http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    unzip -q ADEChallengeData2016.zip
    rm ADEChallengeData2016.zip
fi

# Expected layout:
#   data/ADEChallengeData2016/images/{training,validation}/*.jpg
#   data/ADEChallengeData2016/annotations/{training,validation}/*.png
ls ADEChallengeData2016
