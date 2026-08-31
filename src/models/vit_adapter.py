"""Dense-prediction adapters for the (pre-trained) JEPA vision transformer.

Downstream detection / segmentation need *spatial* feature maps taken from
several intermediate blocks, at resolutions the encoder never saw during
pre-training (512x512 for ADE20K, 1024x1024 for MS-COCO).  This module wraps
`src.models.vision_transformer.VisionTransformer` and provides

- `interpolate_pos_embed_2d`: 2-D (non square capable) positional-embedding
  interpolation, so the frozen/fine-tuned encoder can run at any input size,
- `ViTDenseBackbone`: returns `(B, C, h, w)` maps of the blocks listed in
  `out_layers` (0-indexed, following the paper / mmsegmentation convention:
  layers 3, 5, 7, 11, i.e. the 4th, 6th, 8th and last block of a 12-block ViT),
  with optional ViTDet-style windowed attention to keep the attention cost
  tractable at high resolution,
- `ViTMultiScaleNeck`: the standard ViT feature-pyramid construction that
  resamples those single-stride maps to strides 4/8/16/32.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def interpolate_pos_embed_2d(pos_embed: torch.Tensor, hw: Sequence[int]) -> torch.Tensor:
    """Interpolate a (1, N, D) patch positional embedding onto a (h, w) grid.

    The pre-training positional embedding is a square sin-cos grid without a
    class-token entry (see `VisionTransformer.__init__`), hence `N = g * g`.
    """
    h, w = hw
    n, dim = pos_embed.shape[1], pos_embed.shape[2]
    g = int(round(n ** 0.5))
    assert g * g == n, f"expected a square positional embedding, got {n} tokens"
    if g == h and g == w:
        return pos_embed
    pe = pos_embed.reshape(1, g, g, dim).permute(0, 3, 1, 2)
    pe = F.interpolate(pe.float(), size=(h, w), mode="bicubic", align_corners=False)
    pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, dim)
    return pe.to(pos_embed.dtype)


def window_partition(x: torch.Tensor, hw: Sequence[int], window_size: int):
    """(B, h*w, C) -> (B * num_windows, window_size**2, C), zero padded."""
    h, w = hw
    b, _, c = x.shape
    x = x.view(b, h, w, c)
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, c)
    return x, (hp, wp)


def window_unpartition(x: torch.Tensor, window_size: int, pad_hw: Sequence[int], hw: Sequence[int]):
    """Inverse of `window_partition`."""
    hp, wp = pad_hw
    h, w = hw
    c = x.shape[-1]
    b = x.shape[0] // ((hp // window_size) * (wp // window_size))
    x = x.view(b, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x.view(b, h * w, c)


class ViTDenseBackbone(nn.Module):
    """Wrap a `VisionTransformer` so it emits multi-block spatial feature maps.

    Only the patch tokens are propagated -- the class token carries no spatial
    position and is dropped, which is what ViTDet-style detection/segmentation
    backbones do.

    params:
        vit: pre-trained (target) encoder.
        out_layers: 0-indexed blocks whose outputs are returned; the paper uses
            layers 3, 5, 7 and 11 of a 12-block ViT-B/16 (mmsegmentation
            convention), i.e. its 4th, 6th, 8th and last block.  Blocks after
            the last requested one are never executed.
        window_size: if > 0, all blocks except `global_layers` attend inside
            non-overlapping windows of this many patches (ViTDet-style), which
            is what makes 1024x1024 detection fine-tuning fit in memory.
        global_layers: 0-indexed blocks that keep global attention. Defaults to
            `out_layers`.
        use_checkpoint: gradient checkpointing over the transformer blocks.
        freeze: keep the encoder frozen (linear-probe style evaluation).
    """

    def __init__(
        self,
        vit: nn.Module,
        out_layers: Sequence[int] = (3, 5, 7, 11),
        window_size: int = 0,
        global_layers: Optional[Sequence[int]] = None,
        use_checkpoint: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        self.vit = vit
        depth = len(vit.blocks)
        assert all(0 <= l < depth for l in out_layers), \
            f"out_layers {out_layers} out of range for a {depth}-block encoder"
        self.out_indices = list(out_layers)
        self.last_index = max(self.out_indices)
        self.window_size = window_size
        global_layers = out_layers if global_layers is None else global_layers
        self.global_indices = set(global_layers)
        self.use_checkpoint = use_checkpoint
        self.embed_dim = vit.embed_dim
        self.patch_size = vit.patch_embed.patch_size
        self.num_outputs = len(self.out_indices)
        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        # -- dense prediction runs on the patch tokens only (as in ViTDet): the
        #    class token has no spatial position, so it is dropped and frozen
        #    (an unused trainable parameter would trip DDP).
        if getattr(self.vit, 'cls_token', None) is not None:
            self.vit.cls_token.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()
        return self

    def _run_block(self, blk, x):
        if self.use_checkpoint and self.training:
            return ckpt_fn(blk, x, use_reentrant=False)
        return blk(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, _, h_img, w_img = x.shape
        p = self.patch_size
        assert h_img % p == 0 and w_img % p == 0, \
            f"input {h_img}x{w_img} is not divisible by patch size {p}"
        h, w = h_img // p, w_img // p

        tokens = self.vit.patch_embed(x)                       # (B, h*w, D)
        tokens = tokens + interpolate_pos_embed_2d(self.vit.pos_embed, (h, w))

        outs = []
        for i, blk in enumerate(self.vit.blocks):
            if self.window_size > 0 and i not in self.global_indices:
                shortcut_hw = (h, w)
                win, pad_hw = window_partition(tokens, shortcut_hw, self.window_size)
                win = self._run_block(blk, win)
                tokens = window_unpartition(win, self.window_size, pad_hw, shortcut_hw)
            else:
                tokens = self._run_block(blk, tokens)
            if i in self.out_indices:
                feat = tokens
                if i == len(self.vit.blocks) - 1 and self.vit.norm is not None:
                    feat = self.vit.norm(feat)
                outs.append(feat.transpose(1, 2).reshape(b, -1, h, w).contiguous())
            if i == self.last_index:
                break  # nothing downstream needs the remaining blocks
        return outs


class ViTMultiScaleNeck(nn.Module):
    """Turn single-stride ViT maps into a stride 4/8/16/32 pyramid.

    This is the usual ViT feature-pyramid used by Mask R-CNN / UPerNet on plain
    ViT backbones: the map of each selected block is up/down-sampled by
    `scale_factors`.  When `out_channels` is given, lateral 1x1 + output 3x3
    convolutions project every level to that width (the FPN used for detection);
    otherwise the encoder width is kept (`reduce_dim=False` gives the constant
    768-channel neck used by UPerNet).
    """

    def __init__(
        self,
        embed_dim: int,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        out_channels: Optional[int] = None,
        add_extra_pool: bool = False,
        reduce_dim: bool = True,
    ):
        super().__init__()
        self.scale_factors = list(scale_factors)
        self.add_extra_pool = add_extra_pool
        self.stages = nn.ModuleList()
        self.laterals = nn.ModuleList() if out_channels else None
        self.outputs = nn.ModuleList() if out_channels else None
        dim = embed_dim
        for scale in scale_factors:
            if scale == 4.0:
                mid = dim // 2 if reduce_dim else dim
                cur_dim = dim // 4 if reduce_dim else dim
                layers = [
                    nn.ConvTranspose2d(dim, mid, kernel_size=2, stride=2),
                    LayerNorm2d(mid),
                    nn.GELU(),
                    nn.ConvTranspose2d(mid, cur_dim, kernel_size=2, stride=2),
                ]
            elif scale == 2.0:
                cur_dim = dim // 2 if reduce_dim else dim
                layers = [nn.ConvTranspose2d(dim, cur_dim, kernel_size=2, stride=2)]
            elif scale == 1.0:
                layers = []
                cur_dim = dim
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                cur_dim = dim
            else:
                raise ValueError(f"Unsupported scale factor: {scale}")
            self.stages.append(nn.Sequential(*layers))
            if out_channels is not None:
                self.laterals.append(nn.Sequential(
                    nn.Conv2d(cur_dim, out_channels, kernel_size=1, bias=False),
                    LayerNorm2d(out_channels),
                ))
                self.outputs.append(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(out_channels),
                ))
        self.out_channels = out_channels if out_channels is not None else embed_dim
        if out_channels is not None:
            self.channels_per_level = [out_channels] * len(scale_factors)
        elif reduce_dim:
            self.channels_per_level = [
                embed_dim // 4 if s == 4.0 else embed_dim // 2 if s == 2.0 else embed_dim
                for s in scale_factors
            ]
        else:
            self.channels_per_level = [embed_dim] * len(scale_factors)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats) == len(self.stages), \
            f"expected {len(self.stages)} feature maps, got {len(feats)}"
        outs = []
        for i, (stage, feat) in enumerate(zip(self.stages, feats)):
            out = stage(feat)
            if self.laterals is not None:
                out = self.outputs[i](self.laterals[i](out))
            outs.append(out)
        if self.add_extra_pool:
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return outs
