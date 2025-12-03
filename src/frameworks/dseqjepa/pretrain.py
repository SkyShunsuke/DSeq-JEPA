import argparse
import os
import yaml
import copy
from typing import List
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.cuda import Event as CUDAEvent

from src.utils.masks import get_mask_collator
from src.models import init_model
from src.utils.dist import init_distributed_mode, get_rank, get_world_size
from src.utils.dist import is_main_process as is_main
from src.utils.log import setup_logging, get_logger, AverageMeter
from src.dataset import make_jepa_transforms, make_dataset
from src.utils.opt.optimzer import get_pretrain_optimizer
from src.utils.opt.scheduler import get_cosine_wd_scheduler, get_ema_scheduler, get_warmup_cosine_lr_scheduler, get_lambda_scheduler
from src.utils.opt.scaler import get_gradient_scaler
from src.utils.opt.optimzer import load_checkpoint, save_jepa_checkpoint
from src.eval.feature_extractor import KNNEvaluatorDistributed, build_feature_bank_ddp

from src.frameworks.dseqjepa.attention import SelfGuidedAttention

def logical_or_for_mask(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    params: tensors: List of boolean tensors of the same shape [(B, N), ...]
    returns: torch.Tensor: element-wise logical OR result of shape (B, N)
    """
    if not tensors:
        raise ValueError("Tensor list cannot be empty.")
    stacked = torch.stack(tensors)
    return stacked.amax(dim=0)

def generate_dseq_masks(regions: torch.Tensor):
    """Generate masks for context encoder processing. See Fig. 3. in the paper.
    params: regions: (B, Nd, N) attention-guided binary regions, 1 for selected area, 0 for others
    returns: context_masks: (B*(Nd-1), N), binary masks for context encoder, 0 for keep, 1 for masked
    returns: target_masks: (B*(Nd-1), N), binary masks for target encoder, 1 for loss computation, 0 for ignore
    """
    B, D, N = regions.shape
    masks = []
    prev_masks = []
    for i in range(D-1):
        curr_region = regions[:, i, :]  # (B, N)
        if prev_masks:
            masks.append(logical_or_for_mask(prev_masks))  # (B, N)
        else:
            masks.append(curr_region)
        prev_masks.append(curr_region)
    masks = torch.stack(masks, dim=0)  # ((Nd-1),B, N)
    context_masks = (1 - masks).transpose(0,1).reshape(-1, N)  # (B*(Nd-1), N)
    target_masks = regions[:, 1:, :].reshape(-1, N)  # (B*(Nd-1), N)
    return context_masks, target_masks
    
def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Compute Smooth L1 loss with given mask, which ignores unmasked positions (mask==0).
    params:
        pred: (B, N, D)
        target: (B, N, D)
        mask: (B, N) binary mask
    returns:
        loss: scalar tensor
    """
    if not isinstance(mask, torch.BoolTensor) or not isinstance(mask, torch.cuda.BoolTensor):
        mask = mask.bool()
    
    mask = mask.unsqueeze(-1).expand_as(pred)  # (B, N, D)
    pred_masked = pred[mask].view(-1, pred.size(-1))  # (num_masked, D)
    target_masked = target[mask].view(-1, target.size(-1))  # (num_masked, D)
    loss = F.smooth_l1_loss(pred_masked, target_masked, reduction='mean')
    return loss

def indices_to_binary_mask(masks, N, device=None):
    """
    masks: list of length Nd.
           each m is LongTensor of shape (B, Mp) containing token indices to mask, Mp is a number of masked patches
    N: total token length
    """
    B = masks[0].shape[0]
    binary_masks = []
    for m in masks:
        binary_mask = torch.zeros((B, N), device=device)
        binary_mask.scatter_(1, m, 1)
        binary_masks.append(binary_mask)
    return torch.stack(binary_masks, dim=0)  # (Nd, B, N)

def calculate_masked_ratio(masks: torch.Tensor) -> float:
    """Calculate the average masked ratio from binary masks.
    params: masks: (B*block_num, N) binary masks, 1 for masked, 0 for keep
    returns: float: average masked ratio
    """
    total_elements = masks.numel()
    masked_elements = masks.sum().item()
    return masked_elements / total_elements
    
def main(params, args):
    """Pre-training function for DSeq-JEPA.
    We use following words in the code:
    - encoder: the main JEPA model that encodes the context patch
    - predictor: the prediction head that predicts the target patch representation from the context representation
    - target_encoder: the momentum encoder that encodes the target patch
    - Mp: Number of masked patches, which is shared accross samples in a batch
    - Nenc: Number of context blocks
    - Npred: Number of target blocks
    - Nd: Number of descriminative regions
    - (C, H, W): Channel, Height, Width of the input image
    - (d, h, w): Dimension, Height, Width of the feature map
    - for masking, we assume 0 to be 'keep' and 1 to be 'masked', i.e., each discriminative region has 0s in its area and 1s elsewhere.
    """
    # -- init distributed mode
    init_distributed_mode(args)
    
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:%s'%args.gpu)
    
    # -- setup logging
    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")
    
    # -- make logging stuff
    if is_main():
        log_dir = params['logging']['log_dir']
        use_wandb = params['logging']['wandb'].get('use_wandb', False)
        use_tensorboard = params['logging'].get('use_tensorboard', False)
        use_csv = params['logging'].get('use_csv', False)
        tb_logdir = os.path.join(log_dir, 'tb_logs')
        ckpt_logdir = os.path.join(log_dir, 'checkpoints')
        vis_logdir = os.path.join(log_dir, 'visualizations')
        # this experiment
        os.makedirs(log_dir, exist_ok=True)
        # directory for tensorboard logs
        os.makedirs(tb_logdir, exist_ok=True)
        # directory for saving models
        os.makedirs(ckpt_logdir, exist_ok=True)
        # directory for visualizations
        os.makedirs(vis_logdir, exist_ok=True)
        if use_wandb:
            from src.utils.log import WandbLogger
            wandb_logger = WandbLogger(
                project_name=params['logging']['wandb']['project_name'],
                run_name=params['logging']['wandb']['run_name'],
                entity=params['logging']['wandb']['entity'],
                config=params,
                rank=rank
            )
        if use_tensorboard:
            from src.utils.log import TensorboardLogger
            tensorboard_logger = TensorboardLogger(
                log_dir=tb_logdir,
                rank=rank
            )
        if use_csv:
            from src.utils.log import CSVLogger
            csv_logger = CSVLogger(
                log_dir=log_dir,
                rank=rank
            )
        
        # save config file
        config_save_path = os.path.join(log_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(params, f)    
    else:
        use_csv = False
        use_tensorboard = False
        use_wandb = False
    
    # -- init model
    logger.info("Initializing model...")
    framework_name = params['framework']['name']
    model_name = params['framework']['model']['name']
    model_id = f"{framework_name}_{model_name}"
    patch_size = params['mask']['patch_size']
    crop_size = params['data']['augmentation']['crop_size']
    encoder, predictor = init_model(device, patch_size=patch_size, **params['framework']['model'], crop_size=crop_size, use_masked_vit=True)
    logger.info(f"Model Info: Encoder: {encoder} \n Predictor: {predictor}")
    target_encoder = copy.deepcopy(encoder)
    
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

    _, train_loader, train_sampler = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=train_transform,
        batch_size=params['data']['batch_size_per_replica'],
        collator=mask_collator,
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=world_size,
        rank=rank,
        root_path=params['data']['root_path'],
        data=params['data']['dataset_name'],
    )
    _, train_subset_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
        collator=mask_collator,
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=world_size,
        rank=rank,
        root_path=params['data']['root_path'],
        training=True,
        subset_file=params['data'].get('train_subset_file', None),
        data=params['data']['dataset_name'],
    )
    _, val_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
        collator=None,
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=world_size,
        rank=rank,
        root_path=params['data']['root_path'],
        training=False,
        data=params['data']['dataset_name'],
    )
    ipe = len(train_loader)
    logger.info(f"Data Info: Dataset: {params['data']['dataset_name']}, #Images: {len(train_loader.dataset)}, IPE: {ipe}")
    
    # -- init optimizer and scheduler
    opt_params = params['opt']
    num_epochs = opt_params['epochs']
    optimizer = get_pretrain_optimizer(
        encoder=encoder,
        predictor=predictor,
        optimizer_name=opt_params['name'],
        bias_decay=opt_params['bias_decay'],
        norm_decay=opt_params['norm_decay'],
    )
    lr_scheduler = get_warmup_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=ipe*opt_params['lr_warmup_epochs'],
        start_lr=opt_params['lr_start'],
        ref_lr=opt_params['lr_warmup'],
        T_max=ipe*num_epochs,
        final_lr=opt_params['lr_end'],
        fix_lr_thres=opt_params['fix_lr_thres'],
        fix_strategy=opt_params['fix_lr_strategy'],
    )
    wd_scheduler = get_cosine_wd_scheduler(
        optimizer=optimizer,
        ref_wd=opt_params['wd_start'],
        T_max=ipe*num_epochs,
        final_wd=opt_params['wd_end'],
        fix_wd_thres=opt_params['fix_wd_thres'],
        fix_strategy=opt_params['fix_wd_strategy'],
    )
    ema_scheduler = get_ema_scheduler(
        ema_start=params['framework']['model']['ema_start'],
        ema_end=params['framework']['model']['ema_end'],
        num_epochs=num_epochs,
        iters_per_epoch=ipe,
        ipe_scale=opt_params['ipe_scale']
    )
    lambda_scheduler = get_lambda_scheduler(
        lambda_start=params['framework']['lambda']['lambda_start'],
        lambda_end=params['framework']['lambda']['lambda_end'],
        num_epochs=num_epochs,
        iters_per_epoch=ipe,
        ipe_scale=opt_params['ipe_scale']
    )
    scaler = get_gradient_scaler(
        use_bf16=opt_params['use_bfloat16'],
        device=device
    )
    
    # -- build attention module
    attn_config = params['framework']['attention']
    attn = SelfGuidedAttention(
        strategy=attn_config['strategy'],
        depth=attn_config['depth'],
        alpha=attn_config['alpha'],
        pool_window_size=attn_config['pool_window_size'],
        region_num=attn_config['region_num'],
    )
    
    if args.distributed:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder, static_graph=True)
    
    # -- freeze target encoder params
    for param in target_encoder.parameters():
        param.requires_grad = False
    logger.info("Target encoder parameters frozen.")

    start_epoch = 1
    # -- resume training if needed
    resume_path = params['resume']['resume_path']
    if resume_path is not None:
        assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!, Please check the path."
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            resume_path, encoder, predictor, target_encoder, optimizer, scaler, lr_scheduler, wd_scheduler, ema_scheduler
        )
        logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        logger.info(f"All schedulers and optimizer states are loaded.")
        
    # -- evaluation setup
    eval_strategy = params['logging']['eval']['eval_strategy']
    eval_freq = params['logging']['eval']['eval_epoch_freq']
    logger.info(f"Evaluating with strategy: {eval_strategy} every {eval_freq} epochs.")
    if eval_strategy == 'knn':
        knn = KNNEvaluatorDistributed(
            k=params['logging']['eval']['knn_k'],
            temperature=params['logging']['eval']['knn_t'],
            num_classes=params['logging']['eval']['num_classes'],
            feature_dim=encoder.module.embed_dim,
            feature_strategies=params['logging']['eval']['feature_strategy'],
        )
    else:
        raise NotImplementedError(f"Evaluation strategy {eval_strategy} not implemented.")
    
    logger.info("Starting training... for %d epochs", num_epochs)
    
    # -- initial update for schedulers
    wd_scheduler.step()
    lr_scheduler.step()
    ema_scheduler.step()
    lambda_scheduler.step()
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")
        
        # -- set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        dseq_usage_meter = AverageMeter()
        mask_enc_meter = AverageMeter()
        mask_pred_meter = AverageMeter()
        time_enc_meter = AverageMeter()
        time_pred_meter = AverageMeter()
        
        for step, (batch, masks_enc, masks_pred) in enumerate(train_loader):
            
            # -- load images and masks
            imgs = batch[0].to(device, non_blocking=True)  # (B, C, H, W)
            labels = batch[1].to(device, non_blocking=True)  # (B,)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]  # [(B, Mp) x Nenc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred] # [(B, Mp) x Npred]
            N = imgs.shape[2] // patch_size * imgs.shape[3] // patch_size  # total number of patches

            context_masks = 1 - (
                indices_to_binary_mask(masks_enc, N, device=imgs.device)
                .transpose(1, 2)
                .reshape(-1, N)
            )  # (B*(Nenc), N), 0 for keep, 1 for masked

            target_masks = (
                indices_to_binary_mask(masks_pred, N, device=imgs.device)
                .transpose(1, 2)
                .reshape(-1, N)
            )  # (B*(Npred), N), 1 for loss computation, 0 for ignore
            
            # -- forward pass
            with torch.amp.autocast("cuda", enabled=opt_params['use_bfloat16']):
                
                start, end = CUDAEvent(enable_timing=True), CUDAEvent(enable_timing=True)
                start.record()
                with torch.no_grad():
                    # Since target encoder takes the whole image as input, we also extract informative regions from its internal representation.
                    target_z, attn_z = target_encoder.module.forward_with_intermediate(imgs, blocks=[attn.depth])  
                    target_z = F.layer_norm(target_z, (target_z.size(-1),))  
                end.record()
                torch.cuda.synchronize()
                time_pred_meter.step(start.elapsed_time(end)/1000.0)  # seconds
                
                # sample lambda from bernoulli distribution
                lambda_val = lambda_scheduler.current_lambda()
                use_dseqjepa = (torch.rand(1).item() < lambda_val)
                
                # -- jepa forward
                def jepa_loss(target_z, context_masks, target_masks, imgs):   
                    # -- prediction on random masks 
                    start.record()
                    
                    # Different from the I-JEPA implementation, we perform prediction with masked attention. 
                    # --1. context extraction for context masks
                    context_z = encoder(imgs, masks=context_masks)  # (B*(Nenc), N, d)
                    # --2. target prediction for target masks
                    context_z = context_z.repeat_interleave(len(masks_pred), dim=0)  # (B*(Npred), N, d)
                    context_masks = context_masks.repeat_interleave(len(masks_pred), dim=0)  # (B*(Npred), N)
                    target_z_pred = predictor(context_z, context_masks, target_masks)  # (B*(Npred), N, d)
                    # --3. compute masked loss
                    target_z = target_z.repeat_interleave(len(masks_pred), dim=0)  # (B*(Npred), N, d)
                    loss = masked_smooth_l1_loss(target_z_pred, target_z, target_masks).mean()

                    end.record()
                    torch.cuda.synchronize()
                    time_enc_meter.step(start.elapsed_time(end)/1000.0)  # seconds
                    # -- mask meter ste
                    mask_enc_meter.step(calculate_masked_ratio(context_masks))
                    mask_pred_meter.step(calculate_masked_ratio(target_masks))                    
                    return loss

                # -- dseqjepa loss
                def dseqjepa_loss(target_z, attn_z, imgs):
                    # -- extract attention-guided regions
                    region_query = target_encoder.module.cls_token.expand(imgs.size(0), -1, -1)  # (B, 1, d)
                    regions = attn(attn_z[0], region_query)  # (B, Nd, N)
                    context_masks, target_masks = generate_dseq_masks(regions)  # (B*(Nd-1), N)
                    # -- prediction on attention-guided regions in sequential manner
                    start.record()
                    
                    # Different from I-JEPA, we perform prediction with masked attention. 
                    # This incur additional computation, but leads to better performance.
                    
                    # --1. context extraction for Top-{1,2,...,Nd-1} regions
                    imgs = imgs.repeat_interleave(attn.region_num - 1, dim=0)  # (B*(Nd-1), C, H, W)
                    context_z = encoder(imgs, masks=context_masks)  # (B*(Nd-1), N, d)
                    # --2. target prediction for Top-{2,3,...,Nd} regions
                    target_z_pred = predictor(context_z, context_masks, target_masks)  # (B*(Nd-1), N, d)
                    # --3. compute masked loss
                    target_z = target_z.repeat_interleave(attn.region_num - 1, dim=0)  # (B*(Nd-1), N, d)
                    loss = masked_smooth_l1_loss(target_z_pred, target_z, target_masks).mean()

                    end.record()
                    torch.cuda.synchronize()
                    time_enc_meter.step(start.elapsed_time(end)/1000.0)  # seconds
                    
                    # -- mask meter step
                    mask_enc_meter.step(calculate_masked_ratio(context_masks))
                    mask_pred_meter.step(calculate_masked_ratio(target_masks))
            
                    return loss
                    
                if use_dseqjepa:
                    loss = dseqjepa_loss(target_z, attn_z, imgs)
                else:
                    loss = jepa_loss(target_z, context_masks, target_masks, imgs)
                
                dseq_usage_meter.step(float(use_dseqjepa))
                
            # -- backward pass
            if opt_params['use_bfloat16']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if opt_params['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt_params['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if opt_params['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt_params['clip_grad'])
                optimizer.step()
            optimizer.zero_grad()

            # -- update target encoder
            m = ema_scheduler.current_ema()
            with torch.no_grad():
                for param_q, param_k in zip(encoder.module.parameters(), target_encoder.module.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            
            # -- update meters
            loss = loss.item()
            loss_meter.step(loss)

            # -- log info
            log_step = epoch * ipe + step
            if step % params['logging']['log_step_freq'] == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step}/{ipe}] Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                            f"Mask_enc: {mask_enc_meter.avg} Mask_pred: {mask_pred_meter.avg} "
                            f"Time_enc: {time_enc_meter.avg:.4f}s ({time_enc_meter.avg:.4f}s) "
                            f"Time_pred: {time_pred_meter.avg:.4f}s ({time_pred_meter.avg:.4f}s)"
                            f"Lambda: {lambda_scheduler.current_lambda():.4f} "
                            f"LR: {lr_scheduler.get_current_lr():.6f} WD: {wd_scheduler.get_current_wd():.6f} EMA: {ema_scheduler.current_ema():.4f} "
                            f"DSeq-JEPA Usage: {dseq_usage_meter.avg:.4f}"
                            )
                
                # csv log
                if use_csv:
                    log_dict = {
                        'epoch': epoch,
                        'step': log_step,
                        'loss': loss_meter.val,
                        'loss_avg': loss_meter.avg,
                        'mask_enc': mask_enc_meter.avg,
                        'mask_pred': mask_pred_meter.avg,
                        'time_enc': time_enc_meter.avg,
                        'time_enc_avg': time_enc_meter.avg,
                        'time_pred': time_pred_meter.avg,
                        'time_pred_avg': time_pred_meter.avg,
                        'lr': lr_scheduler.get_current_lr(),
                        'wd': wd_scheduler.get_current_wd(),
                        'ema_m': ema_scheduler.current_ema(),
                    }
                    csv_logger.log_metrics(log_dict, step=log_step)
                
                # tensorboard log
                if use_tensorboard:
                    tb_log_dict = {
                        'Loss/step': loss_meter.val,
                        'Loss/avg': loss_meter.avg,
                        'Mask/mask_enc': mask_enc_meter.avg,
                        'Mask/mask_pred': mask_pred_meter.avg,
                        'Time/time_enc_step': time_enc_meter.avg,
                        'Time/time_enc_avg': time_enc_meter.avg,
                        'Time/time_pred_step': time_pred_meter.avg,
                        'Time/time_pred_avg': time_pred_meter.avg,
                        'LR/lr': lr_scheduler.get_current_lr(),
                        'WD/wd': wd_scheduler.get_current_wd(),
                        'EMA/momentum': ema_scheduler.current_ema(),
                    }
                    tensorboard_logger.log_metrics(tb_log_dict, step=log_step)
                
                # wandb log
                if use_wandb:
                    wandb_log_dict = {
                        'Loss/step': loss_meter.val,
                        'Loss/avg': loss_meter.avg,
                        'Mask/mask_enc': mask_enc_meter.avg,
                        'Mask/mask_pred': mask_pred_meter.avg,
                        'Time/time_enc_step': time_enc_meter.avg,
                        'Time/time_enc_avg': time_enc_meter.avg,
                        'Time/time_pred_step': time_pred_meter.avg,
                        'Time/time_pred_avg': time_pred_meter.avg,
                        'LR/lr': lr_scheduler.get_current_lr(),
                        'WD/wd': wd_scheduler.get_current_wd(),
                        'EMA/momentum': ema_scheduler.current_ema(),
                    }
                    wandb_logger.log_metrics(wandb_log_dict, step=log_step)

            # -- step schedulers
            lr_scheduler.step()
            wd_scheduler.step()
            ema_scheduler.step()
            
        # -- evaluate
        if epoch % eval_freq == 0:
            if eval_strategy == 'knn':
                bank_feats, bank_labels = build_feature_bank_ddp(
                    target_encoder.module,
                    loader=train_subset_loader,
                    normalize=True,
                    use_amp=opt_params['use_bfloat16'],
                    feature_strategies=knn.feature_strategies,
                )
                for strategy in bank_feats.keys():
                    knn.set_bank(bank_feats[strategy], bank_labels[strategy], strategy)
                results_dict = knn.evaluate_ddp(
                    model=target_encoder.module,
                    loader=val_loader,
                    normalize=True,
                    sim_chunk_size=params['logging']['eval'].get('sim_chunk_size', 4096),
                    use_amp=opt_params['use_bfloat16'],
                )
                for strategy in bank_feats.keys():
                    logger.info(f"Epoch [{epoch}/{num_epochs}] KNN Eval Strategy: {strategy} Top-1 Acc: {results_dict[strategy]['top1']:.2f}%, Top-5 Acc: {results_dict[strategy]['top5']:.2f}%")
                    
                    # tensorboard log
                    if use_tensorboard:
                        tb_eval_log_dict = {
                            f'KNN/{strategy}_Top1': results_dict[strategy]['top1'],
                            f'KNN/{strategy}_Top5': results_dict[strategy]['top5'],
                        }
                        tensorboard_logger.log_metrics(tb_eval_log_dict, step=log_step)
                    
                    # wandb log
                    if use_wandb:
                        wandb_eval_log_dict = {
                            f'KNN/{strategy}_Top1': results_dict[strategy]['top1'],
                            f'KNN/{strategy}_Top5': results_dict[strategy]['top5'],
                        }
                        wandb_logger.log_metrics(wandb_eval_log_dict, step=log_step)
            else:
                raise NotImplementedError(f"Evaluation strategy {eval_strategy} not implemented.")

        # -- save checkpoint
        if epoch % params['logging']['ckpt']['ckpt_epoch_freq'] == 0 and is_main():
            save_jepa_checkpoint(
                save_path=os.path.join(ckpt_logdir, f'{model_id}_epoch{epoch}.pth'),
                epoch=epoch,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                ema_scheduler=ema_scheduler,
            )
        elif params['logging']['ckpt']['save_latest'] and is_main():
            save_jepa_checkpoint(
                save_path=os.path.join(ckpt_logdir, f'{model_id}_latest.pth'),
                epoch=epoch,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                ema_scheduler=ema_scheduler,
            )
        else:
            pass
    
    # -- close loggers
    if is_main():
        if use_wandb:
            wandb_logger.close()
        if use_tensorboard:
            tensorboard_logger.close()
        if use_csv:
            csv_logger.close()

    # -- close distributed process
    dist.barrier()
    dist.destroy_process_group()

    # -- end of main
    logger.info("Training completed. Training results are saved in %s", log_dir)