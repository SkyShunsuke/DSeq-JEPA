"""UPerNet semantic-segmentation decoder for plain ViT backbones (ADE20K).

Follows the standard UPerNet recipe used for ViT/BEiT-style encoders: the
feature maps of blocks 3, 5, 7 and 11 are resampled to strides 4/8/16/32 by
`ViTMultiScaleNeck`, a Pyramid Pooling Module is applied to the deepest level,
the levels are fused top-down and concatenated, and a 1x1 convolution predicts
the 150 ADE20K classes.  An auxiliary FCN head on the stride-16 level is
attached during training (weight 0.4), as in the reference implementation.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vit_adapter import ViTDenseBackbone, ViTMultiScaleNeck


def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),  # converted to SyncBatchNorm by the training script
        nn.ReLU(inplace=True),
    )


class PPM(nn.Module):
    """Pyramid Pooling Module (PSPNet)."""

    def __init__(self, in_channels: int, channels: int, pool_scales: Sequence[int] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s), conv_bn_relu(in_channels, channels, 1, 0))
            for s in pool_scales
        ])
        self.bottleneck = conv_bn_relu(in_channels + len(pool_scales) * channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        for stage in self.stages:
            out = stage(x)
            outs.append(F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(outs, dim=1))


class UPerHead(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 512,
        num_classes: int = 150,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channels, pool_scales)
        self.lateral_convs = nn.ModuleList(
            [conv_bn_relu(c, channels, 1, 0) for c in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [conv_bn_relu(channels, channels) for _ in in_channels[:-1]]
        )
        self.bottleneck = conv_bn_relu(len(in_channels) * channels, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats[:-1])]
        laterals.append(self.ppm(feats[-1]))

        # -- top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        outs.append(laterals[-1])

        # -- fuse to the finest resolution
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(outs[i], size=outs[0].shape[2:], mode="bilinear", align_corners=False)
        out = self.bottleneck(torch.cat(outs, dim=1))
        return self.cls_seg(self.dropout(out))


class FCNHead(nn.Module):
    """Auxiliary head (1 conv + classifier)."""

    def __init__(self, in_channels: int, channels: int = 256, num_classes: int = 150, dropout: float = 0.1):
        super().__init__()
        self.conv = conv_bn_relu(in_channels, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_seg(self.dropout(self.conv(x)))


class ViTUPerNet(nn.Module):
    """ViT encoder + multi-scale neck + UPerNet decoder (+ auxiliary FCN head).

    `forward` returns the logits upsampled to the input resolution; during
    training it additionally returns the auxiliary logits.
    """

    def __init__(
        self,
        backbone: ViTDenseBackbone,
        num_classes: int = 150,
        channels: int = 512,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout: float = 0.1,
        use_aux_head: bool = True,
        aux_index: int = 2,
        aux_channels: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = ViTMultiScaleNeck(backbone.embed_dim, (4.0, 2.0, 1.0, 0.5), reduce_dim=False)
        self.decode_head = UPerHead(
            in_channels=self.neck.channels_per_level,
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            dropout=dropout,
        )
        self.use_aux_head = use_aux_head
        self.aux_index = aux_index
        if use_aux_head:
            self.auxiliary_head = FCNHead(
                self.neck.channels_per_level[aux_index], aux_channels, num_classes, dropout
            )

    def forward(self, x: torch.Tensor):
        size = x.shape[2:]
        feats = self.neck(self.backbone(x))
        logits = F.interpolate(self.decode_head(feats), size=size, mode="bilinear", align_corners=False)
        if self.training and self.use_aux_head:
            aux = F.interpolate(
                self.auxiliary_head(feats[self.aux_index]), size=size, mode="bilinear", align_corners=False
            )
            return logits, aux
        return logits
