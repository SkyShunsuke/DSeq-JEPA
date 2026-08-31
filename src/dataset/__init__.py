from src.dataset.transforms import make_jepa_transforms, make_probing_transforms, \
    make_lowlevel_transforms, make_inverse_normalize
from src.dataset.imagenet1k import make_imagenet1k
from src.dataset.fgvc import make_cub200, make_stanford_cars, make_inaturalist21
from src.dataset.clevr import make_clevr
from src.dataset.coco import make_coco, collate_detection
from src.dataset.ade20k import make_ade20k, collate_segmentation


__all__ = [
    "make_jepa_transforms",
    "make_probing_transforms",
    "make_lowlevel_transforms",
    "make_inverse_normalize",
    "make_imagenet1k",
    "make_cub200",
    "make_stanford_cars",
    "make_inaturalist21",
    "make_clevr",
    "make_coco",
    "make_ade20k",
    "collate_detection",
    "collate_segmentation",
    "make_dataset",
]

# -- dataset name (as written in `data.dataset_name`) -> loader factory
DATASET_FACTORIES = {
    "imagenet1k": make_imagenet1k,
    "cub200": make_cub200,
    "stanford_cars": make_stanford_cars,
    "inaturalist21": make_inaturalist21,
    "clevr": make_clevr,
    "coco": make_coco,
    "ade20k": make_ade20k,
}

# -- number of classes of every supported dataset, for config sanity checks
NUM_CLASSES = {
    "imagenet1k": 1000,
    "cub200": 200,
    "stanford_cars": 196,
    "inaturalist21": 10000,
    "coco": 91,       # 80 things + background, original COCO id range
    "ade20k": 150,
}


def make_dataset(
    dataset_name: str,
    **kwargs
) -> tuple:
    """
    Factory method to create dataset and dataloader based on dataset name.
    param: dataset_name: Name of the dataset to create.
    return: (dataset, dataloader, sampler)
    """
    key = dataset_name.lower()
    if key not in DATASET_FACTORIES:
        raise ValueError(f"Dataset {dataset_name} not supported. "
                         f"Available: {sorted(DATASET_FACTORIES)}")
    return DATASET_FACTORIES[key](**kwargs)
