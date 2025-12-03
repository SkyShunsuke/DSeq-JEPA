import os
import torch

from src.utils.masks import get_mask_collator
from src.models import init_target_encoder
from src.utils.log import setup_logging, get_logger
from src.dataset import make_probing_transforms, make_dataset, make_inverse_normalize
from src.utils.opt.optimzer import load_jepa_target_encoder_weights
from src.vis.utils import concat_results_dict
from src.vis.plotting import plot_masking

def main(params, args):
    """Visualization function for JEPA masking with target encoder.
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
    
    # -- make data & transforms
    logger.info("Initializing data...")
    mask_collator = get_mask_collator(**params['mask'], input_size=crop_size)
    mask_info = mask_collator.get_info()
    logger.info(f"Mask Info: {mask_info}")
    
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
        collator=mask_collator,
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
        collator=mask_collator,
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
    
    assert params['data']['batch_size_per_replica'] % num_samples == 0, "Please ensure batch_size_per_replica is divisible by num_samples for visualization."
    num_batches = num_samples // params['data']['batch_size_per_replica']
    current_batch_num = 0
    train_dict = {'masks_enc': [], 'masks_pred': [], 'imgs': torch.tensor([])}
    for (batch, masks_enc, masks_pred) in train_loader:
        imgs = batch[0]
        
        # masks_enc: list of tensors shape (B, N_masked) x num_masks_enc
        # masks_pred: list of tensors shape (B, N_masked) x num_masks_pred
        masks_enc = torch.stack(masks_enc, dim=0).permute(1, 0, 2)  # (B, num_masks_enc, N_masked)
        masks_pred = torch.stack(masks_pred, dim=0).permute(1, 0, 2)  # (B, num_masks_pred, N_masked)
         
        # NOTE: N_masked is varied accross batches. 
        train_dict['imgs'] = torch.cat((train_dict['imgs'], imgs), dim=0)
        train_dict['masks_enc'].append(masks_enc)
        train_dict['masks_pred'].append(masks_pred)
        
        current_batch_num += 1
        if current_batch_num >= num_batches:
            break
    
    imgs = train_dict['imgs']
    masks_enc = train_dict['masks_enc']  # NB x (B, num_masks_enc, N_masked)
    masks_pred = train_dict['masks_pred']  # NB x (B, num_masks_pred, N_masked)
    
    plot_masking(
        imgs,  # (NB * B, C, H, W)
        masks_enc,  # [B, num_masks_enc, N_masked] x NB
        masks_pred, # [B, num_masks_pred, N_masked] x NB
        inv_normalize,
        os.path.join(log_dir, f'train_masking_{num_samples}.png'),
        verbose,
        **vis_config,
    )
    
    current_batch_num = 0
    val_dict = {'masks_enc': [], 'masks_pred': [], 'imgs': torch.tensor([])}
    for (batch, masks_enc, masks_pred) in val_loader:
        imgs = batch[0]
        
        # masks_enc: list of tensors shape (B, N_masked) x num_masks_enc
        # masks_pred: list of tensors shape (B, N_masked) x num_masks_pred
        masks_enc = torch.stack(masks_enc, dim=0).permute(1, 0, 2)  # (B, num_masks_enc, N_masked)
        masks_pred = torch.stack(masks_pred, dim=0).permute(1, 0, 2)  # (B, num_masks_pred, N_masked)
         
        val_dict['imgs'] = torch.cat((val_dict['imgs'], imgs), dim=0)
        val_dict['masks_enc'].append(masks_enc)
        val_dict['masks_pred'].append(masks_pred)
        
        current_batch_num += 1
        if current_batch_num >= num_batches:
            break
    imgs = val_dict['imgs']
    masks_enc = val_dict['masks_enc']  # NB x (B, num_masks_enc, N_masked)
    masks_pred = val_dict['masks_pred']  # NB x (B, num_masks_pred, N_masked)
    
    plot_masking(
        imgs, # (NB * B, C, H, W)
        masks_enc,  # NB x (B, num_masks_enc, N_masked)
        masks_pred, # NB x (B, num_masks_pred, N_masked)
        inv_normalize,
        os.path.join(log_dir, f'val_masking_{num_samples}.png'),
        verbose,
        **vis_config,
    )
    
    # -- save config as yaml
    import yaml
    config_save_path = os.path.join(log_dir, 'visualization_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(params['visualization'], f)
    logger.info(f"Saved visualization config to {config_save_path}")   
    
    # -- end of main
    logger.info("Visualization completed. Visualization results are saved in %s", log_dir)