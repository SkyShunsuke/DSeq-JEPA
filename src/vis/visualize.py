import os
import torch

from src.models import init_target_encoder
from src.utils.log import setup_logging, get_logger
from src.dataset import make_probing_transforms, make_dataset, make_inverse_normalize
from src.utils.opt.optimzer import load_jepa_target_encoder_weights

def main(params, args):
    """Visualization function for JEPA target encoder.
    We use following words in the code:
    - target_encoder: the momentum encoder that encodes the target patch
    - (C, H, W): Channel, Height, Width of the input image
    - (d, h, w): Dimension, Height, Width of the feature map
    """
    device = torch.device(f'cuda')
    
    # -- setup logging
    setup_logging(0, 1)
    logger = get_logger(__name__)
    logger.info(f"Using device: {device}, rank: 0, world_size: 1")
    
    # -- make logging stuff
    log_dir = params['logging']['log_dir']
    # this experiment
    os.makedirs(log_dir, exist_ok=True)
    
    # -- init model
    logger.info("Initializing target encoder...")
    model_name = params['model']['model_name']
    model_id = f"target_encoder_{model_name}"
    patch_size = params['model']['patch_size']
    crop_size = params['data']['augmentation']['crop_size']
    img_size = params['data']['augmentation']['img_size']
    model = init_target_encoder(device, patch_size, model_name, crop_size)
    
    # -- load pre-trained weights into target encoder
    pretrained_weights = params['model']['pretrained_weights']
    assert pretrained_weights is not None, "Please provide pre-trained weights for the target encoder."
    model = load_jepa_target_encoder_weights(
        model,
        pretrained_weights,
        device,
    )
    logger.info(f"Model Info: {model}")
    
    _, test_transform = make_probing_transforms(
        crop_size,
        img_size,
        interpolation=params['data']['augmentation'].get('interpolation', 3),
    )
    inv_normalize = make_inverse_normalize()

    _, train_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=1,
        rank=0,
        subset_file=params['data'].get('train_subset_file', None),
        root_path=params['data']['root_path'],
        data=params['data']['dataset_name'],
    )
    _, val_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=1,
        rank=0,
        root_path=params['data']['root_path'],
        training=False,
        drop_last=False,
        subset_file=params['data'].get('test_subset_file', None),
        data=params['data']['dataset_name'],
    )
    logger.info(f"Data Info: Dataset: {params['data']['dataset_name']}, #Images: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val.")
    model.eval()
    
    # -- run visualization
    verbose = params['logging'].get('verbose', False)
    num_samples = params['visualization']['num_samples']
    vis_strategy = params['visualization']['vis_strategy']
    vis_config = params['visualization']['vis_config']
    logger.info(f"Starting Visualization evaluation for {vis_strategy}...")
    logger.info(f"Visualization config: {vis_config}")
    if vis_strategy == 'attention':
        from src.vis.attention import visualize_attention
        visualize_attention(
            model,
            train_loader,
            val_loader,
            inv_normalize,
            device,
            log_dir,
            num_samples,
            vis_config,
            verbose,
        )
    elif vis_strategy == 'similarity':
        from src.vis.similarity import visualize_similarity
        visualize_similarity(
            model,
            train_loader,
            val_loader,
            inv_normalize,
            device,
            log_dir,
            num_samples,
            vis_config,
            verbose,
        )
    elif vis_strategy == 'clustering':
        from src.vis.clustering import visualize_clustering
        visualize_clustering(
            model,
            train_loader,
            val_loader,
            inv_normalize,
            device,
            log_dir,
            num_samples,
            vis_config,
            verbose,
        )
    else:
        raise ValueError(f"Unsupported visualization strategy: {vis_strategy}")

    # -- end of main
    logger.info("Evaluation completed. Evaluation results are saved in %s", log_dir)