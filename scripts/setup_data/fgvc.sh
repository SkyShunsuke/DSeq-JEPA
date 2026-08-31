#!/bin/bash
# Fine-grained visual categorization datasets: CUB-200-2011 and Stanford-Cars.
# (iNaturalist21 is handled by scripts/setup_data/inaturalist.sh.)
set -euo pipefail

DATA_ROOT=${1:-./data}
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

# -- CUB-200-2011 (200 bird species, official train/test split)
if [ ! -d CUB_200_2011 ]; then
    wget -c https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
    tar -xzf CUB_200_2011.tgz
    rm -f CUB_200_2011.tgz attributes.txt
fi

# -- Stanford-Cars (196 models). The original Stanford URLs are offline, so the
#    archive has to be fetched manually (e.g. from the Kaggle mirror) and placed
#    under data/stanford_cars.  Both layouts below are supported:
#       stanford_cars/{train,test}/<class_name>/*.jpg
#       stanford_cars/{cars_train,cars_test}/*.jpg + devkit/cars_train_annos.mat
#                                                  + cars_test_annos_withlabels.mat
if [ ! -d stanford_cars ]; then
    echo "[stanford_cars] Please download Stanford-Cars manually into $DATA_ROOT/stanford_cars"
    echo "                e.g. kaggle datasets download -d jessicali9530/stanford-cars-dataset"
fi

ls -d CUB_200_2011 stanford_cars 2>/dev/null || true
