"""MS-COCO 2017 detection / instance segmentation loader.

Targets follow the torchvision detection convention (`boxes`, `labels`,
`masks`, `image_id`, `area`, `iscrowd`), and the original COCO category ids
(1..90) are kept so that predictions can be scored directly with COCOeval.
Resizing / normalization / padding is performed by the detector's own
`GeneralizedRCNNTransform`, so the dataset only applies photometric-free
geometric augmentation (random horizontal flip).

Expected layout under `data.root_path`:

    <root>/coco/{train2017,val2017}/*.jpg
    <root>/coco/annotations/instances_{train2017,val2017}.json
"""

import os
import random
from logging import getLogger

import torch
import torchvision
import torchvision.transforms.functional as TF

logger = getLogger()

COCO_NUM_CLASSES = 91  # 80 things + background, keeping the original id range


def collate_detection(batch):
    return tuple(zip(*batch))


def _convert_coco_segm_to_mask(segmentations, height, width):
    """Decode COCO segmentations (polygons, or RLE for crowd regions) to masks."""
    from pycocotools import mask as coco_mask

    masks = []
    for seg in segmentations:
        if isinstance(seg, list):                       # polygons
            rle = coco_mask.merge(coco_mask.frPyObjects(seg, height, width))
        elif isinstance(seg["counts"], list):           # uncompressed RLE
            rle = coco_mask.frPyObjects(seg, height, width)
        else:                                           # compressed RLE
            rle = seg
        mask = coco_mask.decode(rle)
        if mask.ndim > 2:
            mask = mask.any(axis=2)
        masks.append(torch.as_tensor(mask, dtype=torch.uint8))
    if masks:
        return torch.stack(masks, dim=0)
    return torch.zeros((0, height, width), dtype=torch.uint8)


class CocoDetectionDataset(torchvision.datasets.CocoDetection):
    """CocoDetection returning torchvision-style detection targets."""

    def __init__(self, img_folder, ann_file, train=True, hflip_prob=0.5):
        super().__init__(img_folder, ann_file)
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        # -- drop images without any usable annotation from the training set
        if train:
            self.ids = [i for i in self.ids if self._has_valid_annotation(i)]

    def _has_valid_annotation(self, img_id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id, iscrowd=False))
        return any(ann.get("iscrowd", 0) == 0 and ann["bbox"][2] > 1 and ann["bbox"][3] > 1
                   for ann in anns)

    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        image_id = self.ids[index]
        w, h = img.size

        anns = [a for a in anns if a.get("iscrowd", 0) == 0]
        boxes = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]                      # xywh -> xyxy
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)
        labels = torch.as_tensor([a["category_id"] for a in anns], dtype=torch.int64)
        masks = _convert_coco_segm_to_mask([a["segmentation"] for a in anns], h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes, labels, masks = boxes[keep], labels[keep], masks[keep]

        img = TF.to_tensor(img)
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            img = img.flip(-1)
            masks = masks.flip(-1)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + \
                torch.tensor([w, 0, w, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor([a["area"] for a in anns], dtype=torch.float32)[keep],
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return img, target


def make_coco(
    transform=None,
    batch_size=2,
    collator=collate_detection,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder="coco",
    training=True,
    drop_last=True,
    hflip_prob=0.5,
    **kwargs,
):
    """`transform` is ignored: detection preprocessing lives in the detector."""
    split = "train2017" if training else "val2017"
    root = os.path.join(root_path, image_folder)
    dataset = CocoDetectionDataset(
        img_folder=os.path.join(root, split),
        ann_file=os.path.join(root, "annotations", f"instances_{split}.json"),
        train=training,
        hflip_prob=hflip_prob,
    )
    logger.info(f"MS-COCO {split} created: {len(dataset)} images")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=training)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator if collator is not None else collate_detection,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    return dataset, data_loader, dist_sampler
