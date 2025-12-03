import os
import torch
import torchvision.transforms as transforms
import yaml
from src.utils.tensor.kernel import GaussianBlur

from src.models.vision_transformer import VisionTransformer
from src.utils.log import get_logger
from src.vis.plotting import plot_similarity_maps
from src.vis.utils import concat_results_dict

QUERY_TYPES = ['cls', 'gap']
SIM_METRICS = ['cosine', 'l2']
LAYER_AGGREGATION_METHODS = ['each', 'mean']

logger = get_logger(__name__)

def check_methods(
    layer_agg: str,
    sim_metric: str,
    query_type: str,
):
    assert layer_agg in LAYER_AGGREGATION_METHODS, f"Unsupported layer aggregation method: {layer_agg}, supported methods are: {LAYER_AGGREGATION_METHODS}"
    assert sim_metric in SIM_METRICS, f"Unsupported similarity metric: {sim_metric}, supported methods are: {SIM_METRICS}"
    assert query_type in QUERY_TYPES, f"Unsupported query type: {query_type}, supported methods are: {QUERY_TYPES}"

@torch.no_grad()
def get_similarity_maps(
    model: VisionTransformer,
    images: torch.Tensor,
    feature_res: tuple = (14, 14),
    layers: list = [-1],
    query_type: str = 'gap',
    sim_metric: str = 'cosine',
    layer_agg: str = 'each',
    native_sim: bool = False,
    **kwargs
):
    """Get similarity maps from specified layers of the Vision Transformer.
    Args:
        model: VisionTransformer, the model to extract similarity maps from.
        images: torch.Tensor, input images of shape (B, C, H, W).
        feature_res: tuple of int, (h, w) resolution of the feature map.
        layers: list of int, indices of layers to extract similarity from.
        query_type: str, method to get query token ('cls' or 'gap').
        sim_metric: str, similarity metric to use ('cosine' or 'l2').
        layer_agg: str, method to aggregate similarity across layers
                    ('each', 'mean').
        native_sim: bool, whether to return native similarity maps.
    Returns:
        sims_dict: dict, layer index -> similarity maps of shape (B, h, w) or
                    {'mean_sims': (B,h,w)} or {'rollout_sims': (B,h,w)}. 
                    if native_sim is True: return (B, N, N) similarity matrices.
        imgs: torch.Tensor, input images (B, C, H, W).
    """
    assert layers is not None and len(layers) > 0, "Please provide at least one layer index to extract features from."
    if query_type == 'cls':
        assert model.use_class_token, "Model does not use class token, cannot use 'cls' token aggregation."
    elif model.use_class_token:
        logger.info(f"Ignoring class token for token aggregation method: {query_type}")
    else:
        pass

    check_methods(layer_agg=layer_agg, sim_metric=sim_metric, query_type=query_type)
    sims_dict = {}

    # -- prepare tokens
    B = images.size(0)
    x = model.prepare_tokens(images)  # (B, N, D)

    # -- forward pass through transformer blocks
    if native_sim:
        query_type = "none"
        layer_agg = "each"
    
    for i, blk in enumerate(model.blocks):
        x = blk(x, return_attention=False)  # x: (B, N, D)

        if i in layers:
            # -- compute similarity matrix
            if query_type == 'cls':
                query = x[:, 0:1, :]          # (B, 1, D)
                keys   = x[:, 1:, :]          # (B, N-1, D)
            elif query_type == 'gap':
                query = x.mean(dim=1, keepdim=True)  # (B, 1, D)
                keys   = x                      # (B, N, D)
            elif query_type == 'none':
                query = x                      # (B, N, D)
                keys   = x                      # (B, N, D)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")

            # -- compute similarity
            if sim_metric == 'cosine':
                if native_sim:
                    query = query / query.norm(dim=-1, keepdim=True)
                    keys = keys / keys.norm(dim=-1, keepdim=True)
                    sim_map = torch.bmm(query, keys.transpose(1, 2))  # (B, N, N)
                else:
                    sim_map = torch.nn.functional.cosine_similarity(query, keys, dim=-1)  # (B, N-1) or (B, N) or (B, N, N)
            elif sim_metric == 'l2':
                if native_sim:
                    query = query.unsqueeze(2)  # (B, N, 1, D)
                    keys = keys.unsqueeze(1)    # (B, 1, N, D)
                    sim_map = -torch.norm(query - keys, dim=-1)  # (B, N, N)
                else:
                    sim_map = -torch.norm(query - keys, dim=-1)  # (B, N-1) or (B, N) or (B, N, N)
            else:
                raise ValueError(f"Unsupported similarity metric: {sim_metric}")

            # -- reshape to feature map size
            if not native_sim:
                h, w = feature_res
                sim_map = sim_map.reshape(B, h, w)  # (B, h, w)

            # -- store in dict
            sims_dict[f"layer_{i}"] = sim_map

    # -- layer aggregation
    if layer_agg == 'mean':
        sims = list(sims_dict.values())
        sims = sum(sims) / len(sims)  # (B, h, w)
        sims_dict = {'mean_sims': sims.detach().cpu()}
    elif layer_agg == 'each':
        for k in sims_dict:
            sims_dict[k] = sims_dict[k].detach().cpu()
    else:
        raise ValueError(f"Unsupported layer aggregation method: {layer_agg}")

    return sims_dict, images.cpu()

def save_similarity_visualization(
    train_dict: dict,
    val_dict: dict,
    inv_normalize: transforms.Normalize,
    save_dir: str,
    vis_config: dict,
    verbose: bool,
):  
    # -- save train attention maps
    train_save_path = os.path.join(save_dir, 'train_similarity_visualization.png')
    plot_similarity_maps(
        train_dict['imgs'],
        train_dict['sims'],
        inv_normalize,
        train_save_path,
        verbose,
        cmap=vis_config.get('cmap', 'viridis'),
        inverse_score=vis_config.get('inverse_score', False)
    )
    logger.info(f"Saved train similarity visualization to {train_save_path}")
    
    # -- save val attention maps
    val_save_path = os.path.join(save_dir, 'val_similarity_visualization.png')
    plot_similarity_maps(
        val_dict['imgs'],
        val_dict['sims'],
        inv_normalize,
        val_save_path,
        verbose,
        cmap=vis_config.get('cmap', 'viridis'),
        inverse_score=vis_config.get('inverse_score', False)
    )
    
    # -- save config as yaml
    config_save_path = os.path.join(save_dir, 'similarity_visualization_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vis_config, f)
    
    logger.info(f"Saved val similarity visualization to {val_save_path}")
    
def visualize_similarity(
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
        vis_config.get('sim_metric', 'cosine'),
        vis_config.get('query_type', 'gap'),
    )
    
    # -- visualization for train samples
    current_num = 0
    
    train_dict = {'sims': {}, 'imgs': torch.tensor([])}
    for batch in train_loader:
        images, _ = batch
        images = images.to(device, non_blocking=True)
        
        sims_dict, imgs = get_similarity_maps(model, images, **vis_config)  # dict: layer -> similarity maps
        train_dict['sims'] = concat_results_dict(train_dict['sims'], sims_dict)
        train_dict['imgs'] = torch.cat([train_dict['imgs'], imgs], dim=0)
        
        current_num += images.size(0)
        if current_num >= num_samples:
            break
    assert current_num >= num_samples, "Not enough samples in train_loader for visualization."
    
    for k in train_dict['sims']:   
        train_dict['sims'][k] = train_dict['sims'][k][:num_samples]  # (N, h, w)
    train_dict['imgs'] = train_dict['imgs'][:num_samples]  # (N, C, H, W)
    
    # -- visualization for val samples
    current_num = 0
    val_dict = {'sims': {}, 'imgs': torch.tensor([])}
    for batch in val_loader:
        images, _ = batch
        images = images.to(device, non_blocking=True)
        
        sims_dict, imgs = get_similarity_maps(model, images, **vis_config)  # dict
        
        val_dict['sims'] = concat_results_dict(val_dict['sims'], sims_dict)
        val_dict['imgs'] = torch.cat([val_dict['imgs'], imgs], dim=0)
        
        current_num += images.size(0)
        if current_num >= num_samples:
            break
    assert current_num >= num_samples, "Not enough samples in val_loader for visualization."
    for k in val_dict['sims']:   
        val_dict['sims'][k] = val_dict['sims'][k][:num_samples]  # (N, h, w)
    val_dict['imgs'] = val_dict['imgs'][:num_samples]  # (N, C, H, W)
        
    # -- save visualization results
    save_dir = os.path.join(log_dir, 'similarity_visualization')
    os.makedirs(save_dir, exist_ok=True)
    save_similarity_visualization(
        train_dict,
        val_dict,
        inv_normalize,
        save_dir,
        vis_config,
        verbose,
    )
    