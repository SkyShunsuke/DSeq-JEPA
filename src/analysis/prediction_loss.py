import argparse
import os
import yaml
import copy
import torch
import torch.nn.functional as F

from src.utils.masks import get_mask_collator
from src.models import init_model
from src.utils.dist import init_distributed_mode, get_rank, get_world_size
from src.utils.dist import is_main_process as is_main
from src.utils.log import setup_logging, get_logger, AverageMeter
from src.dataset import make_jepa_transforms, make_dataset
from src.utils.opt.optimzer import load_jepa_all_weights_for_analysis

def main(params, args):
    """Loss prediction function for JEPA.
    We use following words in the code:
    - encoder: the main JEPA model that encodes the context patch
    - predictor: the prediction head that predicts the target patch representation from the context representation
    - target_encoder: the momentum encoder that encodes the target patch
    - Mp: Number of masked patches, which is shared accross samples in a batch
    - Nenc: Number of context blocks
    - Npred: Number of target blocks
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
    logger.info("Initializing model...")
    framework_name = params['framework']['name']
    model_name = params['framework']['model']['name']
    model_id = f"{framework_name}_{model_name}"
    patch_size = params['mask']['patch_size']
    crop_size = params['data']['augmentation']['crop_size']
    encoder, predictor = init_model(device, patch_size=patch_size, **params['framework']['model'], crop_size=crop_size)
    logger.info(f"Model Info: Encoder: {encoder} \n Predictor: {predictor}")
    target_encoder = copy.deepcopy(encoder)
    
    # -- load pre-trained weights into target encoder
    pretrained_weights = params['framework']['model']['pretrained_weights']
    assert pretrained_weights is not None, "Please provide pre-trained weights for the target encoder."
    encoder, predictor, target_encoder = load_jepa_all_weights_for_analysis(
        encoder,
        predictor,
        target_encoder,
        pretrained_weights,
        device,
    )
    logger.info("Pre-trained weights loaded into encoder, predictor, and target encoder.")
    
    # -- make data & transforms
    logger.info("Initializing data...")
    mask_collator = get_mask_collator(**params['mask'], input_size=crop_size)
    mask_info = mask_collator.get_info()
    logger.info(f"Mask Info: {mask_info}")
    
    train_transform, test_transform = make_jepa_transforms(
        crop_size=crop_size,
        crop_scale=params['data']['augmentation']['crop_scale'],
        color_jitter=params['data']['augmentation']['color_jitter_strength'],
        horizontal_flip=params['data']['augmentation']['use_horizontal_flip'],
        color_distortion=params['data']['augmentation']['use_color_distortion'],
        gaussian_blur=params['data']['augmentation']['use_gaussian_blur'],
    )

    _, train_subset_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
        collator=mask_collator,
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=1,
        rank=0,
        root_path=params['data']['root_path'],
        training=True,
        subset_file=params['data'].get('train_subset_file', None),
        data=params['data']['dataset_name'],
    )
    _, val_subset_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
        collator=None,
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=1,
        rank=0,
        root_path=params['data']['root_path'],
        training=False,
        subset_file=params['data'].get('test_subset_file', None),
        data=params['data']['dataset_name'],
    )
    logger.info(f"Data Info: Dataset: {params['data']['dataset_name']}, #Images: {len(train_subset_loader.dataset)}")

    
    logger.info("Starting analysis on loss prediction...")
        
    loss_meters = [AverageMeter() for _ in range(4)]
        
    if params["logging"]["verbose"]:
        from tqdm import tqdm
        indices = tqdm(train_subset_loader)
    else:
        indices = train_subset_loader
    for batch, masks_enc, masks_pred in indices:
        # -- load images and masks
        imgs = batch[0].to(device, non_blocking=True)  # (B, C, H, W)
        masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]  # [(B, Mp) x Nenc]
        masks_pred = [m.to(device, non_blocking=True) for m in masks_pred] # [(B, Mp) x Npred]
        n_enc = len(masks_enc)
        n_pred = len(masks_pred)
            
        # -- forward pass
        with torch.amp.autocast("cuda", enabled=params['opt']['use_bfloat16']):
            # -- 1. encode target
            with torch.no_grad():
                target_z = target_encoder(imgs, K=params['framework']['target_depth']) 
                target_z = F.layer_norm(target_z, (target_z.size(-1),))  
                
                # extract target block patches
                target_dim = target_z.size(-1)
                target_z_masked = []
                for m in masks_pred:
                    m_keep = m.unsqueeze(-1).repeat(1, 1, target_dim)  # (B, Mp, d)
                    target_z_masked += [torch.gather(target_z, dim=1, index=m_keep)]
                target_z_masked = torch.cat(target_z_masked, dim=0)  # (B*Npred, Mp, d)
                target_z = target_z_masked.repeat_interleave(len(masks_enc), dim=0)  # (B*Npred*Nenc, Mp, d)
            
            # -- 2. encode context & predict target
            context_z = encoder(imgs, masks_enc)
            target_z_pred = predictor(context_z, masks_enc, masks_pred)
            
            # -- 3. compute loss
            pred_loss = F.smooth_l1_loss(target_z_pred, target_z, reduction='none').mean(dim=(1, 2)) # (B*Npred*Nenc, )

        # -- update each loss meter
        pred_loss = pred_loss.view(-1, n_pred, n_enc)  # (B, Npred, Nenc)
        pred_loss = pred_loss.mean(dim=0)  # (Npred, Nenc)
        
        for i in range(len(loss_meters)):
            loss = pred_loss[i][0].detach().cpu().item()
            loss_meters[i].step(loss)
        
        break
    
    # -- log each loss averaged values
    for i in range(len(loss_meters)):
        avg_loss = loss_meters[i].avg
        logger.info(f"Loss for block [{i}]: {avg_loss:.3f}")

    # -- end of main
    logger.info("Analysis completed. Analysis results are saved in %s", log_dir)

    