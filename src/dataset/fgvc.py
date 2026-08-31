"""Fine-grained visual categorization datasets: iNaturalist21, CUB-200-2011, Stanford-Cars.

All three are evaluated with exactly the same frozen-encoder linear-probing
protocol as ImageNet (see `src/eval/linear_probing.py`), so the loaders follow
the `make_imagenet1k` contract and simply return `(image, label)` pairs.

Expected layouts under `data.root_path`:

    <root>/CUB_200_2011/{images.txt,image_class_labels.txt,train_test_split.txt,images/}
    <root>/stanford_cars/{train,test}/<class_name>/*.jpg          # folder-style mirrors
    <root>/stanford_cars/{cars_train,cars_test}/*.jpg             # original release
        + devkit/cars_train_annos.mat, cars_test_annos_withlabels.mat
    <root>/inaturalist21/{2021_train_mini,2021_valid}/            # torchvision layout
"""

import os
from logging import getLogger
from typing import Callable, Optional

import torch
import torchvision
from PIL import Image

logger = getLogger()

CUB_NUM_CLASSES = 200
CARS_NUM_CLASSES = 196
INAT21_NUM_CLASSES = 10000


def _make_loader(dataset, batch_size, collator, pin_mem, num_workers, world_size, rank, drop_last):
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    return data_loader, dist_sampler


class CUB200(torch.utils.data.Dataset):
    """CUB-200-2011 with the official train/test split."""

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        img_dir = os.path.join(root, "images")
        assert os.path.isdir(img_dir), f"CUB images not found under {img_dir}"

        with open(os.path.join(root, "images.txt")) as f:
            id_to_path = dict(line.split() for line in f.read().splitlines())
        with open(os.path.join(root, "image_class_labels.txt")) as f:
            id_to_label = dict(line.split() for line in f.read().splitlines())
        with open(os.path.join(root, "train_test_split.txt")) as f:
            id_to_split = dict(line.split() for line in f.read().splitlines())

        self.samples = [
            (os.path.join(img_dir, id_to_path[i]), int(id_to_label[i]) - 1)  # labels are 1-indexed
            for i in id_to_path
            if (id_to_split[i] == "1") == train
        ]
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class StanfordCars(torch.utils.data.Dataset):
    """Stanford-Cars, either from the original devkit annotations or a folder mirror."""

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        split = "train" if train else "test"

        folder_style = os.path.join(root, split)
        if os.path.isdir(folder_style):
            base = torchvision.datasets.ImageFolder(folder_style)
            self.samples = list(base.samples)
        else:
            from scipy.io import loadmat  # only needed for the original release
            img_dir = os.path.join(root, f"cars_{split}")
            anno = os.path.join(root, "devkit", "cars_train_annos.mat") if train else \
                os.path.join(root, "cars_test_annos_withlabels.mat")
            if not os.path.isfile(anno):  # some mirrors keep both files in devkit/
                anno = os.path.join(root, "devkit", os.path.basename(anno))
            assert os.path.isdir(img_dir) and os.path.isfile(anno), \
                f"Stanford-Cars {split} split not found under {root}"
            annos = loadmat(anno, squeeze_me=True)["annotations"]
            self.samples = [
                (os.path.join(img_dir, str(a["fname"])), int(a["class"]) - 1) for a in annos
            ]
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_cub200(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="CUB_200_2011",
    training=True,
    drop_last=True,
    **kwargs,
):
    dataset = CUB200(os.path.join(root_path, image_folder), train=training, transform=transform)
    logger.info(f"CUB-200-2011 ({'train' if training else 'test'}) created: {len(dataset)} images")
    loader, sampler = _make_loader(dataset, batch_size, collator, pin_mem, num_workers,
                                   world_size, rank, drop_last)
    return dataset, loader, sampler


def make_stanford_cars(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="stanford_cars",
    training=True,
    drop_last=True,
    **kwargs,
):
    dataset = StanfordCars(os.path.join(root_path, image_folder), train=training, transform=transform)
    logger.info(f"Stanford-Cars ({'train' if training else 'test'}) created: {len(dataset)} images")
    loader, sampler = _make_loader(dataset, batch_size, collator, pin_mem, num_workers,
                                   world_size, rank, drop_last)
    return dataset, loader, sampler


def make_inaturalist21(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="inaturalist21",
    training=True,
    drop_last=True,
    train_version="2021_train_mini",
    val_version="2021_valid",
    target_type="full",
    **kwargs,
):
    """iNaturalist 2021 (10,000 species).

    `train_version` selects between the full (`2021_train`, 2.7M images) and the
    mini (`2021_train_mini`, 500K images, 50 per species) training set.
    """
    version = train_version if training else val_version
    dataset = torchvision.datasets.INaturalist(
        root=os.path.join(root_path, image_folder),
        version=version,
        target_type=target_type,
        transform=transform,
        download=False,
    )
    logger.info(f"iNaturalist21 ({version}) created: {len(dataset)} images")
    loader, sampler = _make_loader(dataset, batch_size, collator, pin_mem, num_workers,
                                   world_size, rank, drop_last)
    return dataset, loader, sampler
