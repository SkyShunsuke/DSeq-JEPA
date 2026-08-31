# `src/eval` -- downstream evaluations

| Module | Task | Entry point (`main.py`) |
|---|---|---|
| `linear_probing.py` | Linear probing (ImageNet-1K, iNaturalist21, CUB-200-2011, Stanford-Cars) | `--task classification` + `framework.name: probing` |
| `knn.py` | kNN classification | `--task classification` + `framework.name: knn` |
| `low_level.py` | Low-level reasoning: Clevr/Count, Clevr/Dist (VTAB-1k) | `--task classification` + `framework.name: low_level` |
| `detection.py` | MS-COCO detection / instance segmentation (Mask R-CNN) | `--task detection` |
| `segmentation.py` | ADE20K semantic segmentation (UPerNet) | `--task segmentation` |
| `common.py` | Shared distributed / logging / checkpoint plumbing and `build_pretrained_encoder` | -- |
| `feature_extractor.py` | Feature banks used by the kNN evaluator | -- |

Every task loads the pre-trained **target encoder** through
`common.build_pretrained_encoder`, which honours `model.use_masked_vit`,
`model.use_class_token` and `model.crop_size`; these must match the
pre-training config, otherwise the checkpoint will not load.

Dense tasks (`detection.py`, `segmentation.py`) wrap that encoder in
`src/models/vit_adapter.py::ViTDenseBackbone`, which re-interpolates the
positional embedding to the downstream resolution and returns the feature maps
of the blocks listed in `model.out_layers` (0-indexed: `[3, 5, 7, 11]`).
