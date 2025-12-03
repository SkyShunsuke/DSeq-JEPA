"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from src.utils.tensor.distribution import trunc_normal_

from typing import List, Tuple
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(True)        # SM80+, FP16/BF16
torch.backends.cuda.enable_mem_efficient_sdp(True)  # fallback for SM75+

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_class_token=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_class_token = use_class_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop 
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        # qkv: (B, N, 3, H, C//H) -> (3, B, H, N, C//H)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if mask is not None:
            # mask input: (B, N) with 0 for keep, 1 for masked
            if self.use_class_token:
                mask_for_cls = torch.zeros((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask_for_cls, mask], dim=1)

            attn_mask = (mask == 0).bool()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        dropout_p = self.attn_drop_p if self.training else 0.0
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.scale
        )
        attn = None 

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_class_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_class_token=use_class_token)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attention=False):
        norm1 = self.norm1(x)
        
        y, attn = self.attn(norm1, mask=mask)
        
        x = x + self.drop_path(y)
        norm2 = self.norm2(x)
        x = x + self.drop_path(self.mlp(norm2))
        
        if return_attention:
            return x, attn
        else:
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2
        
    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)

class MaskedVisionTransformerPredictor(nn.Module):
    """ Vision Transformer for Predictor with token masking"""
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        apply_stop=False,
        noise_var=0.25,
        learned_pos_emb=False,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        if not learned_pos_emb:
            self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                    requires_grad=False)
            predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                        int(num_patches**.5),
                                                        cls_token=False)
            self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        else:
            self.predictor_pos_embed = nn.Parameter(torch.randn(1, num_patches, predictor_embed_dim) * 0.02, requires_grad=True)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, use_class_token=False)
            for i in range(depth)])

        self.init_std = init_std
        self.apply_stop = apply_stop
        self.noise_var = noise_var

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_enc, context_masks, target_masks, return_attention=False):
        """Masked Predictor.
        params: x_enc: torch.Tensor: Context encoder embeddings of shape (B*(Nd-1), N, D_enc)
        params: context_masks: torch.Tensor: Binary masks for context tokens of shape (B*(Nd-1), N), 0 for keep, 1 for masked
        params: target_masks: torch.Tensor: Binary masks for target tokens of shape (B*(Nd-1), N), 1 for loss computation, 0 for ignore
        returns: torch.Tensor: Predicted target embeddings of shape (B*(Nd-1), N, D_enc)
        """
        assert target_masks is not None, 'Cannot run predictor without binary masks!'
        n_enc_masks = len(context_masks)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x_enc)
        B, N, D = x.shape

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1) # (B, N, D)
        x = x + x_pos_embed

        # -- replace with mask tokens to masked positions
        mask_pos_emb = self.predictor_pos_embed.repeat(B, 1, 1)
        if self.apply_stop:
            noise = torch.normal(mean=0., std=self.noise_var, size=(B, N, D), device=mask_pos_emb.device)
            mask_pos_emb += self.predictor_embed(noise)
        x = (1 - target_masks.unsqueeze(-1).repeat(1, 1, D)) * x + target_masks.unsqueeze(-1).repeat(1, 1, D) * mask_pos_emb
        # -- fwd prop
        # take logical or b/w context and target masks
        context_target_masks = 1 - logical_or_for_mask([context_masks, target_masks])  # (B*(Nd-1), N)
        for blk in self.predictor_blocks:
            blk_out = blk(x, return_attention=return_attention, mask=context_target_masks)
            if return_attention:
                x, attn = blk_out
            else:
                x = blk_out
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = self.predictor_proj(x)
        return x
    
def logical_or_for_mask(masks: List[torch.Tensor]) -> torch.Tensor:
    """Compute logical OR for a list of binary masks.
    params: masks: List[torch.Tensor]: List of binary masks of shape (B, N), 0 for keep, 1 for masked
    returns: torch.Tensor: Combined binary mask of shape (B, N), 0 for keep, 1 for masked
    """
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = torch.clamp(combined_mask + mask, max=1)
    return combined_mask
    
class FeatAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # bs, seq_len, dims = x.shape
        x = x.permute((0, 2, 1))
        return self.avg_pool(x).squeeze()

class MaskedVisionTransformer(nn.Module):
    """ Vision Transformer for Target/Context Encoder"""
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        use_class_token=False,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --  positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.use_class_token = use_class_token
        if self.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=init_std)
        else:
            self.cls_token = None

        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, use_class_token=use_class_token)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.avg_pool = FeatAvgPool()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def prepare_tokens(self, x, masks=None):
        """Target Encoder / Context Encoder token preparation with optional masking.
        params: x: torch.Tensor: Input images of shape (B, C, H, W)
        params: masks: torch.Tensor: Optional binary masks of shape (B*(Nd-1), N), 0 for keep, 1 for masked
        returns: torch.Tensor: Prepared tokens of shape (B*(Nd-1), N_masked, D) or (B, N, D) if no masks provided
        """
        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape
        
        # -- add class token
        if self.use_class_token:
            class_tokens = (
                self.cls_token.expand(B, -1, -1) 
            )
            x = torch.cat((class_tokens, x), dim=1)

        # -- add positional embedding to x
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        if self.use_class_token:
            pos_embed = self.pos_embed.repeat(B, 1, 1)
            pos_embed = torch.cat((torch.zeros(B, 1, D, device=x.device), pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed.repeat(B, 1, 1)
        x = x + pos_embed
        return x
        
    def forward(self, x, masks=None, K=1, return_attention=False):
        """Context/Target Encoder forward pass with optional masking.
        params: x: torch.Tensor: Input images of shape (B, C, H, W)
        params: masks: torch.Tensor: Optional binary masks of shape (B*(Nd-1), N), 0 for keep, 1 for masked
        params: K: int: Number of last blocks to average for output feature
        returns: torch.Tensor: Output features of shape (B*(Nd-1), D) or (B, D) if no masks provided
        """
        x = self.prepare_tokens(x, masks)

        # -- fwd prop
        x_K = []
        for i, blk in enumerate(self.blocks):
            if return_attention:
                x, _attn = blk(x, return_attention=True, mask=masks)
            else:
                x = blk(x, mask=masks)
            if i >= len(self.blocks) - K:
                _x = x
                if self.norm is not None:
                    _x = self.norm(_x)
                x_K.append(_x)
        x = sum(x_K) / len(x_K)
        
        if self.use_class_token:
            x = x[:, 1:, :]
        return x

    def forward_with_intermediate(self, x, blocks=None, masks=None, K=1, return_attention=False):
        x = self.prepare_tokens(x, masks)

        # -- fwd prop
        x_K = []
        x_b = []
        for i, blk in enumerate(self.blocks):
            if return_attention:
                x, _attn = blk(x, return_attention=True, mask=masks)
            else:
                x = blk(x, mask=masks)
                
            if i >= len(self.blocks) - K:
                _x = x
                if self.norm is not None:
                    _x = self.norm(_x)
                x_K.append(_x)
            if blocks is not None and i in blocks:
                _x = x
                if self.norm is not None:
                    _x = self.norm(_x)
                x_b.append(_x)
        x = sum(x_K) / len(x_K)
        if self.use_class_token:
            x = x[:, 1:, :]
            x_b = [t[:, 1:, :] for t in x_b]
        return x, x_b

    def get_intermediate_features(
        self, x: torch.Tensor, names: List[str], masks=None
    ) -> dict:
        """
        Available feature keys:
        - blkCLS{integer}         => CLS token of blk[{integer}]
        - concatCLS{integer}      => concat of CLS tokens from last {integer} blocks
        - lastCLS                 => CLS token of last block
        - lastMAP                 => feature map (no CLS) of last block [B, H, W, C]
        - lastBLK                 => [CLS | MAP] full sequence of last block
        - concatBLK{integer}      => concat of full sequences from last {integer} blocks
        - lastPOOL                => pooled feature of last block (USE_CLASS_TOKEN=False 前提)
        - concatPOOL{integer}     => concat of pooled features from last {integer} blocks
        - stridePOOL_{cnt}_{str}  => pooled features sampled by stride from end
        """
        interms = []
        x = self.prepare_tokens(x, masks)
        for blk in self.blocks:
            x = blk(x)
            interms.append(self.norm(x[:, 1:, :] if self.use_class_token else x))

        output = {}
        for name in names:
            if name.startswith("blkCLS"):
                assert self.use_class_token, "Need class token to extract blkCLS"
                v = int(name.replace("blkCLS", ""))
                output[name] = interms[v][:, 0]
            elif name.startswith("concatCLS"):
                assert self.use_class_token, "Need class token to extract concatCLS"
                v = int(name.replace("concatCLS", ""))
                feat = torch.cat([t[:, 0] for t in interms[-v:]], dim=-1)
                output[name] = feat
            elif name == "lastCLS":
                assert self.use_class_token, "Need class token to extract lastCLS"
                output[name] = interms[-1][:, 0]
            elif name == "lastMAP":
                feat_map_size = self.patch_embed.feat_map_size
                if self.use_class_token:
                    feat_map = interms[-1][:, 1:]
                else:
                    feat_map = interms[-1]
                B, L, C = feat_map.shape
                feat_map = feat_map.reshape((B, *feat_map_size, C))
                output[name] = feat_map
            elif name == "lastBLK":
                output[name] = interms[-1]
            elif name.startswith("concatBLK"):
                v = int(name.replace("concatBLK", ""))
                feat = torch.cat(interms[-v:], dim=-1)
                output[name] = feat
            elif name == "lastPOOL":
#                 assert not self.use_class_token, "Pooling with class token not supported"
                output[name] = self.avg_pool(interms[-1])
            elif name.startswith("concatPOOL"):
                # assert not self.use_class_token, "Pooling with class token not supported"
                v = int(name.replace("concatPOOL", ""))
                feat = torch.cat([self.avg_pool(t) for t in interms[-v:]], dim=-1)
                output[name] = feat
            elif name.startswith("stridePOOL"):
                # assert not self.use_class_token, "Pooling with class token not supported"
                name_ = name.replace("stridePOOL_", "")
                parts = [int(s) for s in name_.split("_")]
                if len(parts) == 2:
                    count, stride = parts
                else:
                    count = int(name_)
                    stride, r = divmod(len(interms), count)
                    stride = stride + 1 if r else stride
                feat = torch.cat(
                    [self.avg_pool(t) for t in interms[::-stride][:count]], dim=-1
                )
                output[name] = feat

        return output

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = MaskedVisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = MaskedVisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)