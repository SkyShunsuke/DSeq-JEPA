from src.dataset.transforms import make_jepa_transforms, make_probing_transforms, make_inverse_normalize
from src.dataset.imagenet1k import make_imagenet1k


__all__ = [
    "make_jepa_transforms",
    "make_imagenet1k",
]

def make_dataset(
    dataset_name: str,
    **kwargs
) -> tuple:
    """
    Factory method to create dataset and dataloader based on dataset name.
    param: dataset_name: Name of the dataset to create.
    return: (dataset, dataloader, sampler)
    """
    if dataset_name.lower() == 'imagenet1k':
        return make_imagenet1k(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")