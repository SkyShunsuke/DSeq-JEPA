# `src/dataset` -- datasets and loaders

`make_dataset(dataset_name, **kwargs)` returns `(dataset, dataloader, sampler)`
for every supported dataset:

| `data.dataset_name` | Module | Used by | Expected layout under `data.root_path` |
|---|---|---|---|
| `imagenet1k` | `imagenet1k.py` | pre-training, probing, kNN | `imagenet/{train,val}/<wnid>/*.JPEG` |
| `cub200` | `fgvc.py` | linear probing | `CUB_200_2011/{images.txt,image_class_labels.txt,train_test_split.txt,images/}` |
| `stanford_cars` | `fgvc.py` | linear probing | `stanford_cars/{train,test}/<class>/*.jpg` or the original devkit release |
| `inaturalist21` | `fgvc.py` | linear probing | `inaturalist21/{2021_train_mini,2021_valid}/` |
| `clevr` | `clevr.py` | Clevr/Count, Clevr/Dist | `CLEVR_v1.0/{images,scenes}/` |
| `coco` | `coco.py` | detection / instance segmentation | `coco/{train2017,val2017,annotations}/` |
| `ade20k` | `ade20k.py` | semantic segmentation | `ADEChallengeData2016/{images,annotations}/{training,validation}/` |

`scripts/setup_data/` downloads all of them except Stanford-Cars (whose original
URLs are offline) and ImageNet.

Transforms live in `transforms.py`: `make_jepa_transforms` (pre-training),
`make_probing_transforms` (linear probing / kNN) and `make_lowlevel_transforms`
(VTAB-style resize for the CLEVR tasks).  Detection and segmentation build their
augmentation inside their dataset / detector instead.
