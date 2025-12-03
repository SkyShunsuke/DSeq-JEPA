import os
import torch
import torchvision.transforms as transforms
import yaml
from src.utils.tensor.kernel import GaussianBlur

from src.models.vision_transformer import VisionTransformer
from src.utils.log import get_logger
from src.vis.plotting import plot_attention_maps
from src.vis.utils import concat_results_dict

TOKEN_AGGREGATION_METHODS = ['reception', 'emission', 'cls']
LAYER_AGGREGATION_METHODS = ['rollout', 'each', 'mean']
HEAD_AGGREGATION_METHODS = ['mean', 'max', 'min']

logger = get_logger(__name__)

def check_methods(
    layer_agg: str,
    head_agg: str,
    token_agg: str,
):
    assert layer_agg in LAYER_AGGREGATION_METHODS, f"Unsupported layer aggregation method: {layer_agg}, supported methods are: {LAYER_AGGREGATION_METHODS}"
    assert head_agg in HEAD_AGGREGATION_METHODS, f"Unsupported head aggregation method: {head_agg}, supported methods are: {HEAD_AGGREGATION_METHODS}"
    assert token_agg in TOKEN_AGGREGATION_METHODS, f"Unsupported token aggregation method: {token_agg}, supported methods are: {TOKEN_AGGREGATION_METHODS}"

@torch.no_grad()
def get_attention_maps(
    model: VisionTransformer,
    images: torch.Tensor,
    feature_res: tuple = (14, 14),
    layers: list = [-1],
    token_agg: str = 'reception',
    head_agg: str = 'mean',
    layer_agg: str = 'each',
    native_attn: bool = False,
    **kwargs
):
    """Get attention maps from specified layers of the Vision Transformer.
    Args:
        model: VisionTransformer, the model to extract attention maps from.
        images: torch.Tensor, input images of shape (B, C, H, W).
        feature_res: tuple of int, (h, w) resolution of the feature map.
        layers: list of int, indices of layers to extract attention from.
        token_agg: str, token aggregation method.
        head_agg: str, head aggregation method.
        layer_agg: str, layer aggregation method: {'each','mean','rollout'}.
        native_attn: bool, whether to return native attention maps.
    Returns:
        attns_dict: dict, layer index -> attention maps of shape (B, h, w) or
                    {'mean_attns': (B,h,w)} or {'rollout_attns': (B,h,w)}.
                    if native_attn is True: return (B, N, N) attention matrices.
        imgs: torch.Tensor, input images (B, C, H, W).
    """
    assert layers is not None and len(layers) > 0, "Please provide at least one layer index to extract attention from."
    if token_agg == 'cls':
        assert model.use_class_token, "Model does not use class token, cannot use 'cls' token aggregation."
    elif model.use_class_token:
        logger.info(f"Ignoring class token for token aggregation method: {token_agg}")
    else:
        pass

    check_methods(layer_agg=layer_agg, head_agg=head_agg, token_agg=token_agg)

    attns_dict = {}
    rollout_mats = []  # store (B, N, N) for rollout (after head-agg, before token-agg)

    # -- prepare tokens
    B = images.size(0)
    x = model.prepare_tokens(images)  # (B, N, D)

    if native_attn:
        token_agg = "none"
        layer_agg = "each"
    
    # -- forward pass through transformer blocks
    for i, blk in enumerate(model.blocks):
        x, attn = blk(x, return_attention=True)  # x: (B, N, D), attn: (B, H, N, N)

        if i in layers:
            # -- head aggregation -> (B, N, N)
            if head_agg == 'mean':
                attn = attn.mean(dim=1)
            elif head_agg == 'max':
                attn, _ = attn.max(dim=1)
            elif head_agg == 'min':
                attn, _ = attn.min(dim=1)
            else:
                raise ValueError(f"Unsupported head aggregation method: {head_agg}")

            # keep the square matrix for rollout
            rollout_mats.append(attn)

            # -- token aggregation -> (B, N-1) or (B, N) if no CLS
            if token_agg == 'cls':
                cls_attn = attn[:, 0, :]          # (B, N)
                attn_map = cls_attn[:, 1:]        # (B, N-1)
            elif token_agg == 'reception':
                attn_map = attn[:, 1:, 1:] if model.use_class_token else attn  # (B, N-1, N-1) or (B, N, N)
                attn_map = attn_map.mean(dim=1)   # (B, N-1) or (B, N)
            elif token_agg == 'emission':
                attn_map = attn[:, 1:, 1:] if model.use_class_token else attn
                attn_map = attn_map.mean(dim=2)   # (B, N-1) or (B, N)
            elif token_agg == 'none':
                attn_map = attn                   # (B, N, N)
            else:
                raise ValueError(f"Unsupported token aggregation method: {token_agg}")

            # -- reshape to feature map size
            if not native_attn:
                h, w = feature_res
                attn_map = attn_map.reshape(B, h, w)  # (B, h, w)

            # -- store in dict
            attns_dict[f"layer_{i}"] = attn_map

    # -- layer aggregation
    if layer_agg == 'mean':
        attns = list(attns_dict.values())
        attns = sum(attns) / len(attns)  # (B, h, w)
        attns_dict = {'mean_attns': attns.detach().cpu()}

    elif layer_agg == 'rollout':
        # ---- Attention Rollout (Abnar & Zuidema)
        assert len(rollout_mats) > 0, "No attention matrices collected for rollout."

        # ensure the same order as 'layers' argument (in case model.blocks order vs. insertion differs)
        # rebuild rollout_mats in the provided layer order
        layer_to_mat = {}
        idx = 0
        for i, blk in enumerate(model.blocks):
            if i in layers:
                layer_to_mat[i] = rollout_mats[idx]
                idx += 1
        mats = [layer_to_mat[i] for i in layers]  # [(B,N,N), ...]

        B, N, _ = mats[0].shape
        device = mats[0].device
        eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)

        proc = []
        for A in mats:
            A = A + eye                      # add residual
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)  # row-normalize
            proc.append(A)

        roll = proc[0]
        for A in proc[1:]:
            roll = torch.bmm(A, roll)        # (B, N, N), shallow->deep

        h, w = feature_res
        if model.use_class_token:
            # class-to-patch: take CLS row (or column depending on convention).
            # Here we use roll[:, 0, 1:] meaning influence from CLS to patches.
            cls_to_patches = roll[:, 0, 1:]          # (B, h*w)
            attn_map = cls_to_patches.reshape(B, h, w)
        else:
            # no CLS: aggregate node importance across sources (row-mean)
            patch_scores = roll.mean(dim=1)          # (B, N=h*w)
            attn_map = patch_scores.reshape(B, h, w)

        # normalize per sample (sum=1) for stability/interpretability
        flat = attn_map.view(B, -1)
        flat = flat / (flat.sum(dim=1, keepdim=True) + 1e-8)
        attn_map = flat.view(B, h, w)

        attns_dict = {'rollout_attns': attn_map.detach().cpu()}

    elif layer_agg == 'each':
        for k in attns_dict:
            attns_dict[k] = attns_dict[k].detach().cpu()
    else:
        raise ValueError(f"Unsupported layer aggregation method: {layer_agg}")

    return attns_dict, images.cpu()

def save_attention_visualization(
    train_dict: dict,
    val_dict: dict,
    inv_normalize: transforms.Normalize,
    save_dir: str,
    vis_config: dict,
    verbose: bool,
):  
    # -- save train attention maps
    train_save_path = os.path.join(save_dir, 'train_attention_visualization.png')
    plot_attention_maps(
        train_dict['imgs'],
        train_dict['attns'],
        inv_normalize,
        train_save_path,
        verbose,
        cmap=vis_config.get('cmap', 'viridis'),
        inverse_score=vis_config.get('inverse_score', False)
    )
    logger.info(f"Saved train attention visualization to {train_save_path}")
    
    # -- save val attention maps
    val_save_path = os.path.join(save_dir, 'val_attention_visualization.png')
    plot_attention_maps(
        val_dict['imgs'],
        val_dict['attns'],
        inv_normalize,
        val_save_path,
        verbose,
        cmap=vis_config.get('cmap', 'viridis'),
        inverse_score=vis_config.get('inverse_score', False)
    )
    
    # -- save config as yaml
    config_save_path = os.path.join(save_dir, 'attention_visualization_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vis_config, f)
    
    logger.info(f"Saved val attention visualization to {val_save_path}")
    
def visualize_attention(
    model: VisionTransformer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    inv_normalize: transforms.Normalize,
    device: torch.device,
    log_dir: str,
    num_samples: int,
    vis_config: dict,
    verbose: bool,
):
    assert model.training is False, "Model should be in eval mode for visualization."
    check_methods(
        vis_config.get('layer_agg', 'each'),
        vis_config.get('head_agg', 'mean'),
        vis_config.get('token_agg', 'reception'),
    )
    
    # -- visualization for train samples
    current_num = 0
    
    train_dict = {'attns': {}, 'imgs': torch.tensor([])}
    for batch in train_loader:
        images, _ = batch
        images = images.to(device, non_blocking=True)
        
        attns_dict, imgs = get_attention_maps(model, images, **vis_config)  # dict: layer -> attn maps
        train_dict['attns'] = concat_results_dict(train_dict['attns'], attns_dict)
        train_dict['imgs'] = torch.cat([train_dict['imgs'], imgs], dim=0)
        
        current_num += images.size(0)
        if current_num >= num_samples:
            break
    assert current_num >= num_samples, "Not enough samples in train_loader for visualization."
    
    for k in train_dict['attns']:   
        train_dict['attns'][k] = train_dict['attns'][k][:num_samples]  # (N, h, w)
    train_dict['imgs'] = train_dict['imgs'][:num_samples]  # (N, C, H, W)
    
    # -- visualization for val samples
    current_num = 0
    val_dict = {'attns': {}, 'imgs': torch.tensor([])}
    for batch in val_loader:
        images, _ = batch
        images = images.to(device, non_blocking=True)
        
        attns_dict, imgs = get_attention_maps(model, images, **vis_config)  # dict
        
        val_dict['attns'] = concat_results_dict(val_dict['attns'], attns_dict)
        val_dict['imgs'] = torch.cat([val_dict['imgs'], imgs], dim=0)
        
        current_num += images.size(0)
        if current_num >= num_samples:
            break
    assert current_num >= num_samples, "Not enough samples in val_loader for visualization."
    for k in val_dict['attns']:   
        val_dict['attns'][k] = val_dict['attns'][k][:num_samples]  # (N, h, w)
    val_dict['imgs'] = val_dict['imgs'][:num_samples]  # (N, C, H, W)
        
    # -- save visualization results
    save_dir = os.path.join(log_dir, 'attention_visualization')
    os.makedirs(save_dir, exist_ok=True)
    save_attention_visualization(
        train_dict,
        val_dict,
        inv_normalize,
        save_dir,
        vis_config,
        verbose,
    )
    