import argparse
import os
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.cuda import Event as CUDAEvent

from src.models import init_target_encoder, init_probing_model
from src.utils.dist import init_distributed_mode, get_rank, get_world_size
from src.utils.dist import is_main_process as is_main
from src.utils.log import setup_logging, get_logger, AverageMeter
from src.dataset import make_probing_transforms, make_dataset
from src.utils.opt.optimzer import get_probing_optimizer
from src.utils.opt.scheduler import get_multi_step_values_lr_scheduler
from src.utils.opt.scaler import get_gradient_scaler
from src.utils.opt.optimzer import load_jepa_target_encoder_weights, load_probing_checkpoint, save_probing_checkpoint

def main(params, args):
    """Linear probing function for JEPA target encoder.
    We use following words in the code:
    - target_encoder: the momentum encoder that encodes the target patch
    - head: the linear or MLP classification head for linear probing
    - (C, H, W): Channel, Height, Width of the input image
    - (d, h, w): Dimension, Height, Width of the feature map
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
    else:
        use_csv = False
        use_tensorboard = False
        use_wandb = False
    
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
    
    # -- build linear probing model
    model_with_head = init_probing_model(
        backbone=model,
        freeze_backbone=True,
        embed_dim=model.embed_dim,
        num_classes=params['model']['num_classes'],
        head_type=params['model']['head_type'],
        use_bn=params['model']['use_bn'],
        mlp_config=params['model'].get('mlp_config', None)
    )
    
    logger.info(f"Model Info: {model}")
    
    train_transform, test_transform = make_probing_transforms(
        crop_size,
        img_size,
        interpolation=params['data']['augmentation'].get('interpolation', 3),
    )

    _, train_loader, train_sampler = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=train_transform,
        batch_size=params['data']['batch_size_per_replica'],
        pin_mem=params['data']['pin_memory'],
        num_workers=params['data']['num_workers'],
        world_size=world_size,
        rank=rank,
        subset_file=params['data'].get('train_subset_file', None),
        root_path=params['data']['root_path'],
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
        drop_last=False,
        subset_file=params['data'].get('test_subset_file', None),
        data=params['data']['dataset_name'],
    )
    ipe = len(train_loader)
    logger.info(f"Data Info: Dataset: {params['data']['dataset_name']}, #Images: {len(train_loader.dataset)}, IPE: {ipe}")
    
    # -- init optimizer and scheduler
    opt_params = params['opt']
    num_epochs = opt_params['epochs']
    optimizer = get_probing_optimizer(
        model_with_head,
        optimizer_name=opt_params['name'],
        bias_decay=opt_params['bias_decay'],
        norm_decay=opt_params['norm_decay'],
        weight_decay=opt_params['weight_decay'],
        base_lr=opt_params['lr']['base_lr'],
        world_size=world_size,
        batch_size_per_replica=params['data']['batch_size_per_replica'],
        base_lr_batch_size=opt_params['lr'].get('base_lr_batch_size', 256),
        auto_lr_scaling=opt_params['lr'].get('auto_lr_scaling', True),
    )
    
    assert opt_params['lr']['name'] == 'multistep', "Only multistep LR scheduler is supported for linear probing."
    lr_scheduler = get_multi_step_values_lr_scheduler(
        optimizer=optimizer,
        milestones=opt_params['lr']['milestones'],
        values=opt_params['lr']['lr_values'],
        iterations_per_epoch=ipe,
        start_step=0
    )
    scaler = get_gradient_scaler(
        use_bf16=opt_params['use_bfloat16'],
        device=device
    )
    
    if args.distributed:
        if params['model']['use_bn']:
            logger.info("Using SyncBatchNorm for distributed training.")
            model_with_head = nn.SyncBatchNorm.convert_sync_batchnorm(model_with_head).to(device)
        model_with_head = DistributedDataParallel(model_with_head, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    start_epoch = 1
    # -- resume training if needed
    resume_path = params['resume']['resume_path']
    if resume_path is not None:
        assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!, Please check the path."
        model_with_head, optimizer, scaler, lr_scheduler, start_epoch = load_probing_checkpoint(
            resume_path, model_with_head, optimizer, scaler, lr_scheduler
        )
        logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        logger.info(f"All schedulers and optimizer states are loaded.")
        
    eval_freq = params['logging']['eval']['eval_epoch_freq']
    logger.info(f"Evaluating every {eval_freq} epochs.")
    logger.info("Starting training... for %d epochs", num_epochs)
    model_with_head.train()
    
    # -- main training loop
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")
        
        # -- set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        
        for step, batch in enumerate(train_loader):
            
            # -- load images and labels
            imgs = batch[0].to(device, non_blocking=True)  # (B, C, H, W)
            labels = batch[1].to(device, non_blocking=True)  # (B,)
            
            # -- forward pass
            with torch.amp.autocast("cuda", enabled=opt_params['use_bfloat16']):
                start, end = CUDAEvent(enable_timing=True), CUDAEvent(enable_timing=True)
                start.record()
                out_pool4, out_last = model_with_head(imgs)  # (B, num_classes), (B, num_classes)
                loss_pool4 = F.cross_entropy(out_pool4, labels)
                loss_last = F.cross_entropy(out_last, labels)
                loss = (loss_pool4 + loss_last) / 2.0
            
            end.record()
            torch.cuda.synchronize()
            time_meter.step(start.elapsed_time(end) / 1000.0)  # in seconds
            
            # -- backward pass
            optimizer.zero_grad()
            if opt_params['use_bfloat16']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if opt_params['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(model_with_head.parameters(), opt_params['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if opt_params['clip_grad'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(model_with_head.parameters(), opt_params['clip_grad'])
                optimizer.step()
            
            # -- update meters
            loss = loss.item()
            loss_meter.step(loss)

            # -- log info
            if step % params['logging']['log_step_freq'] == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step}/{ipe}] Loss: {loss_meter.val:.4f} (Avg: {loss_meter.avg:.4f}), Time: {time_meter.val:.4f}s (Avg: {time_meter.avg:.4f}s)")
                
                # csv log
                if use_csv:
                    log_dict = {
                        'epoch': epoch,
                        'step': epoch * ipe + step,
                        'loss': loss_meter.val,
                        'loss_avg': loss_meter.avg,
                        'time_step': time_meter.val,
                        'time_avg': time_meter.avg,
                        'lr': lr_scheduler.get_current_lr(),
                    }
                    csv_logger.log_metrics(log_dict, step=epoch * ipe + step)
                
                # tensorboard log
                if use_tensorboard:
                    tb_log_dict = {
                        'Loss/step': loss_meter.val,
                        'Loss/avg': loss_meter.avg,
                        'Time/time_step': time_meter.val,
                        'Time/time_avg': time_meter.avg,
                        'LR/lr': lr_scheduler.get_current_lr(),
                    }
                    tensorboard_logger.log_metrics(tb_log_dict, step=epoch * ipe + step)
                
                # wandb log
                if use_wandb:
                    wandb_log_dict = {
                        'Loss/step': loss_meter.val,
                        'Loss/avg': loss_meter.avg,
                        'Time/time_step': time_meter.val,
                        'Time/time_avg': time_meter.avg,
                        'LR/lr': lr_scheduler.get_current_lr(),
                    }
                    wandb_logger.log_metrics(wandb_log_dict)

            # -- step schedulers
            lr_scheduler.step()
            
        # -- evaluate
        if (epoch) % eval_freq == 0:
            logger.info("Evaluating linear probing model at epoch %d", epoch)
            model_with_head.eval()
            local_total = 0
            local_correct_pool4 = 0
            local_correct_last = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs_pool4, outputs_last = model_with_head(images)
                    _, predicted_pool4 = torch.max(outputs_pool4, dim=1)
                    _, predicted_last  = torch.max(outputs_last,  dim=1)

                    local_total += labels.size(0)
                    local_correct_pool4 += (predicted_pool4 == labels).sum().item()
                    local_correct_last  += (predicted_last  == labels).sum().item()

            t_total   = torch.tensor([local_total], dtype=torch.long, device=device)
            t_pool4   = torch.tensor([local_correct_pool4], dtype=torch.long, device=device)
            t_last    = torch.tensor([local_correct_last],  dtype=torch.long, device=device)

            if args.distributed:
                dist.all_reduce(t_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_pool4, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_last,  op=dist.ReduceOp.SUM)

            total   = t_total.item()
            correct_pool4 = t_pool4.item()
            correct_last  = t_last.item()

            acc_pool4 = 100.0 * correct_pool4 / max(1, total)
            acc_last  = 100.0 * correct_last  / max(1, total)
            
            logger.info(f"Validation Results - Epoch: {epoch}, Pool4 Accuracy: {acc_pool4:.2f}%, Last Accuracy: {acc_last:.2f}%")
            # log eval results
            if use_tensorboard:
                tb_eval_log_dict = {
                    'Val/Accuracy_Pool4': acc_pool4,
                    'Val/Accuracy_Last': acc_last,
                }
                tensorboard_logger.log_metrics(tb_eval_log_dict, step=epoch * ipe)
            if use_wandb:
                wandb_eval_log_dict = {
                    'Val/Accuracy_Pool4': acc_pool4,
                    'Val/Accuracy_Last': acc_last,
                }
                wandb_logger.log_metrics(wandb_eval_log_dict, step=epoch * ipe)
            model_with_head.train()

        # -- save checkpoint
        if (epoch) % params['logging']['ckpt']['ckpt_epoch_freq'] == 0 and is_main():
            save_probing_checkpoint(
                save_path=os.path.join(ckpt_logdir, f'{model_id}_epoch_{epoch+1}.pth'),
                epoch=epoch,
                model_with_head=model_with_head,
                opt=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
            )
        elif params['logging']['ckpt']['save_latest'] and is_main():
            save_probing_checkpoint(
                save_path=os.path.join(ckpt_logdir, f'{model_id}_latest.pth'),
                epoch=epoch,
                model_with_head=model_with_head,
                opt=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
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

    