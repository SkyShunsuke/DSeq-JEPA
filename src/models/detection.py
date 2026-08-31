"""Mask R-CNN with a plain-ViT + FPN backbone (MS-COCO detection / instance seg).

The pre-trained encoder is turned into a stride 4/8/16/32(/64) feature pyramid
by `ViTDenseBackbone` + `ViTMultiScaleNeck` (blocks 3, 5, 7, 11), which is then
plugged into torchvision's `MaskRCNN`, so RPN / RoI heads / losses / inference
follow the reference Mask R-CNN implementation.
"""

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from src.models.vit_adapter import ViTDenseBackbone, ViTMultiScaleNeck

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViTFPN(nn.Module):
    """ViT backbone + feature pyramid, in the format torchvision detectors expect."""

    def __init__(self, backbone: ViTDenseBackbone, out_channels: int = 256, extra_pool: bool = True):
        super().__init__()
        self.body = backbone
        self.fpn = ViTMultiScaleNeck(
            backbone.embed_dim,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            out_channels=out_channels,
            add_extra_pool=extra_pool,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        feats = self.fpn(self.body(x))
        return OrderedDict((str(i), f) for i, f in enumerate(feats))


def build_mask_rcnn(
    encoder: nn.Module,
    num_classes: int = 91,
    out_layers: Sequence[int] = (3, 5, 7, 11),
    fpn_channels: int = 256,
    img_size: int = 1024,
    window_size: int = 14,
    use_checkpoint: bool = True,
    freeze_backbone: bool = False,
    anchor_sizes: Sequence[int] = (32, 64, 128, 256, 512),
    **kwargs,
) -> MaskRCNN:
    """Assemble Mask R-CNN on top of a pre-trained JEPA encoder.

    `num_classes` counts the background class; COCO category ids are kept in
    their original 1..90 range, hence the default of 91.
    """
    backbone = ViTDenseBackbone(
        encoder,
        out_layers=out_layers,
        window_size=window_size,
        use_checkpoint=use_checkpoint,
        freeze=freeze_backbone,
    )
    body = ViTFPN(backbone, out_channels=fpn_channels, extra_pool=True)

    anchor_generator = AnchorGenerator(
        sizes=tuple((s,) for s in anchor_sizes),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes),
    )
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

    model = MaskRCNN(
        body,
        num_classes=num_classes,
        min_size=img_size,
        max_size=img_size,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        **kwargs,
    )
    # -- ViT needs the batched image size to be a multiple of the patch size, and the
    #    neck halves the token grid once, so it must be a multiple of 2*patch_size too.
    #    Round that up to at least 32 (patch 16 -> 32, unchanged; patch 14 -> 56).
    stride = backbone.patch_size * 2
    model.transform.size_divisible = stride * -(-32 // stride)
    return model
