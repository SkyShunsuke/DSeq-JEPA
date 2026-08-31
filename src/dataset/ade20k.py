"""ADE20K (SceneParse150) semantic-segmentation loader.

Training pipeline follows the standard ADE20K recipe used with UPerNet:
random resize (ratio 0.5-2.0 of the 2048x512 base scale), random crop to
512x512 with a category-balance constraint, random horizontal flip,
photometric distortion, normalization and padding with the ignore label.
Validation images are returned at their original resolution (after an optional
resize of the short side) and scored with sliding-window inference.

Expected layout under `data.root_path`:

    <root>/ADEChallengeData2016/images/{training,validation}/*.jpg
    <root>/ADEChallengeData2016/annotations/{training,validation}/*.png

Annotation pngs store 0 for "unlabelled" and 1..150 for the classes; they are
mapped to 0..149 with 255 as the ignore index.
"""

import os
import random
from logging import getLogger
from typing import Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

logger = getLogger()

ADE20K_NUM_CLASSES = 150
IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ADE20KSegmentation(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        crop_size: int = 512,
        base_size: Sequence[int] = (2048, 512),
        ratio_range: Sequence[float] = (0.5, 2.0),
        cat_max_ratio: float = 0.75,
        photometric_distortion: bool = True,
        test_scale: Optional[Sequence[int]] = (2048, 512),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        split = "training" if train else "validation"
        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        assert os.path.isdir(self.img_dir), f"ADE20K images not found under {self.img_dir}"
        self.names = sorted(n[:-4] for n in os.listdir(self.img_dir) if n.endswith(".jpg"))
        self.train = train
        self.crop_size = crop_size
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.cat_max_ratio = cat_max_ratio
        self.photometric_distortion = photometric_distortion
        self.test_scale = test_scale
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.names)

    # -- augmentation helpers -------------------------------------------------
    def _resize(self, img, ann, size):
        w, h = img.size
        long_side, short_side = max(size), min(size)
        scale = min(long_side / max(w, h), short_side / min(w, h))
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        return img.resize((nw, nh), Image.BILINEAR), ann.resize((nw, nh), Image.NEAREST)

    def _random_crop(self, img, ann):
        crop = self.crop_size
        ann_np = np.array(ann)
        for _ in range(10):
            h, w = ann_np.shape
            y = random.randint(0, max(0, h - crop))
            x = random.randint(0, max(0, w - crop))
            patch = ann_np[y:y + crop, x:x + crop]
            if self.cat_max_ratio < 1.0:
                labels, counts = np.unique(patch[patch != IGNORE_INDEX], return_counts=True)
                if len(labels) > 1 and counts.max() / max(1, counts.sum()) < self.cat_max_ratio:
                    break
            else:
                break
        box = (x, y, x + crop, y + crop)
        return img.crop(box), ann.crop(box)

    def _photometric_distortion(self, img):
        img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
        img = TF.adjust_contrast(img, random.uniform(0.75, 1.25))
        img = TF.adjust_saturation(img, random.uniform(0.75, 1.25))
        img = TF.adjust_hue(img, random.uniform(-0.03, 0.03))
        return img

    # ------------------------------------------------------------------------
    def __getitem__(self, index):
        name = self.names[index]
        img = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        ann = Image.open(os.path.join(self.ann_dir, name + ".png"))

        if self.train:
            ratio = random.uniform(*self.ratio_range)
            size = (int(self.base_size[0] * ratio), int(self.base_size[1] * ratio))
            img, ann = self._resize(img, ann, size)
            ann = Image.fromarray(self._remap(np.array(ann)))
            img, ann = self._random_crop(img, ann)
            if random.random() < 0.5:
                img, ann = TF.hflip(img), TF.hflip(ann)
            if self.photometric_distortion:
                img = self._photometric_distortion(img)
            img = self._normalize(img)
            label = torch.from_numpy(np.array(ann)).long()
            img, label = self._pad(img, label)
        else:
            if self.test_scale is not None:
                img, ann = self._resize(img, ann, self.test_scale)
            img = self._normalize(img)
            label = torch.from_numpy(self._remap(np.array(ann))).long()
        return img, label

    def _remap(self, ann: np.ndarray) -> np.ndarray:
        ann = ann.astype(np.int32) - 1  # 0 = unlabelled -> ignore
        ann[ann < 0] = IGNORE_INDEX
        return ann.astype(np.uint8)

    def _normalize(self, img):
        return TF.normalize(TF.to_tensor(img), self.mean, self.std)

    def _pad(self, img, label):
        crop = self.crop_size
        ph, pw = crop - img.shape[1], crop - img.shape[2]
        if ph > 0 or pw > 0:
            img = torch.nn.functional.pad(img, (0, max(0, pw), 0, max(0, ph)), value=0.0)
            label = torch.nn.functional.pad(
                label, (0, max(0, pw), 0, max(0, ph)), value=IGNORE_INDEX)
        return img, label


def collate_segmentation(batch):
    """Validation images keep their own size, so they are returned as lists."""
    imgs, labels = zip(*batch)
    if all(i.shape == imgs[0].shape for i in imgs):
        return torch.stack(imgs, 0), torch.stack(labels, 0)
    return list(imgs), list(labels)


def make_ade20k(
    transform=None,
    batch_size=2,
    collator=collate_segmentation,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="ADEChallengeData2016",
    training=True,
    drop_last=True,
    crop_size=512,
    base_size=(2048, 512),
    ratio_range=(0.5, 2.0),
    cat_max_ratio=0.75,
    photometric_distortion=True,
    test_scale=(2048, 512),
    **kwargs,
):
    """`transform` is ignored: the segmentation pipeline is built into the dataset."""
    dataset = ADE20KSegmentation(
        root=os.path.join(root_path, image_folder),
        train=training,
        crop_size=crop_size,
        base_size=tuple(base_size),
        ratio_range=tuple(ratio_range),
        cat_max_ratio=cat_max_ratio,
        photometric_distortion=photometric_distortion,
        test_scale=tuple(test_scale) if test_scale is not None else None,
    )
    logger.info(f"ADE20K ({'training' if training else 'validation'}) created: {len(dataset)} images")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=training)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator if collator is not None else collate_segmentation,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    return dataset, data_loader, dist_sampler
