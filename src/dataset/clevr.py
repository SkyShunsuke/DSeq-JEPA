"""CLEVR low-level reasoning tasks (Clevr/Count and Clevr/Dist) in the VTAB-1k setting.

Two classification problems are derived from the CLEVR scene graphs, exactly as
in VTAB:

- `count`     : number of objects in the scene, 3..10  -> 8 classes.
- `distance`  : depth of the closest object, binned with the VTAB thresholds
                [0, 8.0, 8.5, 9.0, 9.5, 10.0] -> 6 classes.

VTAB-1k splits: the first 800 images of the (90%) train split are used for
training, the first 200 images of the held-out 10% for validation, and the
15,000 images of the official CLEVR validation set for testing.

Expected layout under `data.root_path`:

    <root>/CLEVR_v1.0/images/{train,val}/*.png
    <root>/CLEVR_v1.0/scenes/CLEVR_{train,val}_scenes.json
"""

import json
import os
from logging import getLogger
from typing import Callable, Optional

import torch
from PIL import Image

logger = getLogger()

COUNT_MIN_OBJECTS = 3
COUNT_NUM_CLASSES = 8
DISTANCE_THRESHOLDS = [0.0, 8.0, 8.5, 9.0, 9.5, 10.0]
DISTANCE_NUM_CLASSES = len(DISTANCE_THRESHOLDS)

VTAB_TRAIN_PERCENT = 90  # VTAB keeps the last 10% of the train split for validation
VTAB_TRAIN_SIZE = 800
VTAB_VAL_SIZE = 200


def _count_label(scene: dict) -> int:
    label = len(scene["objects"]) - COUNT_MIN_OBJECTS
    assert 0 <= label < COUNT_NUM_CLASSES, f"unexpected object count {len(scene['objects'])}"
    return label


def _distance_label(scene: dict) -> int:
    # pixel_coords = (x, y, z); z is the distance of the object to the camera.
    dist = min(obj["pixel_coords"][2] for obj in scene["objects"])
    return max(i for i, t in enumerate(DISTANCE_THRESHOLDS) if t - dist < 0)


LABEL_FNS = {"count": _count_label, "distance": _distance_label}
NUM_CLASSES = {"count": COUNT_NUM_CLASSES, "distance": DISTANCE_NUM_CLASSES}


class CLEVRClassification(torch.utils.data.Dataset):
    """CLEVR images with count / closest-distance labels and VTAB-1k splits."""

    def __init__(
        self,
        root: str,
        task: str = "count",
        split: str = "train800",
        transform: Optional[Callable] = None,
    ):
        assert task in LABEL_FNS, f"task must be one of {list(LABEL_FNS)}, got {task}"
        assert split in ("train800", "val200", "train800val200", "trainval", "test"), \
            f"unsupported split {split}"
        self.root = root
        self.task = task
        self.split = split
        self.transform = transform
        self.num_classes = NUM_CLASSES[task]

        source = "val" if split == "test" else "train"
        scene_file = os.path.join(root, "scenes", f"CLEVR_{source}_scenes.json")
        assert os.path.isfile(scene_file), f"CLEVR scenes not found: {scene_file}"
        with open(scene_file) as f:
            scenes = json.load(f)["scenes"]
        scenes = sorted(scenes, key=lambda s: s["image_filename"])

        label_fn = LABEL_FNS[task]
        img_dir = os.path.join(root, "images", source)
        items = [(os.path.join(img_dir, s["image_filename"]), label_fn(s)) for s in scenes]

        if split == "test":
            self.samples = items
        else:
            n_train = len(items) * VTAB_TRAIN_PERCENT // 100
            train_pool, val_pool = items[:n_train], items[n_train:]
            if split == "train800":
                self.samples = train_pool[:VTAB_TRAIN_SIZE]
            elif split == "val200":
                self.samples = val_pool[:VTAB_VAL_SIZE]
            elif split == "train800val200":
                self.samples = train_pool[:VTAB_TRAIN_SIZE] + val_pool[:VTAB_VAL_SIZE]
            else:  # trainval: the full 90%/10% splits (non 1k regime)
                self.samples = train_pool

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_clevr(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="CLEVR_v1.0",
    training=True,
    drop_last=True,
    task="count",
    train_split="train800",
    test_split="test",
    **kwargs,
):
    dataset = CLEVRClassification(
        root=os.path.join(root_path, image_folder),
        task=task,
        split=train_split if training else test_split,
        transform=transform,
    )
    logger.info(f"CLEVR/{task} ({dataset.split}) created: {len(dataset)} images, "
                f"{dataset.num_classes} classes")
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
    return dataset, data_loader, dist_sampler
