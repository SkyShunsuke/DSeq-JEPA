#!/bin/bash
# MS-COCO 2017 -- object detection and instance segmentation transfer.
set -euo pipefail

DATA_ROOT=${1:-./data}
DEST="$DATA_ROOT/coco"
mkdir -p "$DEST"
cd "$DEST"

for f in train2017 val2017; do
    if [ ! -d "$f" ]; then
        wget -c "http://images.cocodataset.org/zips/$f.zip"
        unzip -q "$f.zip" && rm "$f.zip"
    fi
done
if [ ! -d annotations ]; then
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip
fi

# Expected layout:
#   data/coco/{train2017,val2017}/*.jpg
#   data/coco/annotations/instances_{train2017,val2017}.json
ls -d train2017 val2017 annotations
