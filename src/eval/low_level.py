"""Low-level reasoning transfer: Clevr/Count and Clevr/Dist (VTAB-1k).

Protocol (paper, Sec. "Downstream Evaluation Details"):
the target encoder is frozen and a task-specific linear classifier is trained on
the final-layer [CLS] token of 224x224 inputs -- 8-way for Clevr/Count, 6-way for
Clevr/Dist -- with AdamW (lr 1e-3, weight decay 0.05), batch size 256, for 100
epochs with a cosine schedule, on the VTAB-1k split (800 train / 200 val /
15,000 test).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.dataset import make_dataset, make_lowlevel_transforms
from src.dataset.clevr import NUM_CLASSES as CLEVR_NUM_CLASSES
from src.eval.common import setup_experiment, build_pretrained_encoder, all_reduce_sum, teardown
from src.models import init_low_level_model
from src.utils.distributed import is_main_process as is_main
from src.utils.log import AverageMeter
from src.utils.opt.optimzer import get_finetune_optimizer, save_downstream_checkpoint, \
    load_downstream_checkpoint
from src.utils.opt.scaler import get_gradient_scaler
from src.utils.opt.scheduler import get_warmup_cosine_lr_scheduler


@torch.no_grad()
def evaluate(model, loader, device, use_bfloat16=False):
    """Top-1 accuracy, reduced over all ranks."""
    model.eval()
    correct = torch.zeros(1, dtype=torch.long, device=device)
    total = torch.zeros(1, dtype=torch.long, device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_bfloat16):
            logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum()
        total += labels.numel()
    all_reduce_sum(correct)
    all_reduce_sum(total)
    model.train()
    return 100.0 * correct.item() / max(1, total.item())


def main(params, args):
    run = setup_experiment(params, args)
    device, rank, world_size = run['device'], run['rank'], run['world_size']
    logger, loggers, ckpt_dir = run['logger'], run['loggers'], run['ckpt_dir']

    data_params = params['data']
    task = data_params.get('task', 'count')
    num_classes = CLEVR_NUM_CLASSES[task]
    crop_size = data_params['augmentation']['crop_size']
    logger.info(f"CLEVR/{task}: {num_classes}-way classification, {crop_size}x{crop_size} inputs")

    # -- frozen pre-trained encoder + task-specific head
    encoder = build_pretrained_encoder(params, device, img_size=crop_size)
    feature_key = params['model'].get('feature_key', 'lastCLS')
    if feature_key.endswith('CLS'):
        assert encoder.use_class_token, \
            "feature_key requires a class token; set model.use_class_token: true"
    model = init_low_level_model(
        backbone=encoder,
        freeze_backbone=params['model'].get('freeze_backbone', True),
        embed_dim=encoder.embed_dim,
        num_classes=num_classes,
        feature_key=feature_key,
        head_type=params['model'].get('head_type', 'linear'),
        use_bn=params['model'].get('use_bn', False),
        mlp_config=params['model'].get('mlp_config', None),
    ).to(device)

    # -- data (VTAB-1k splits)
    train_transform, test_transform = make_lowlevel_transforms(crop_size)
    common = dict(
        dataset_name=data_params['dataset_name'],
        batch_size=data_params['batch_size_per_replica'],
        pin_mem=data_params['pin_memory'],
        num_workers=data_params['num_workers'],
        world_size=world_size,
        rank=rank,
        root_path=data_params['root_path'],
        task=task,
    )
    _, train_loader, train_sampler = make_dataset(
        transform=train_transform, training=True, drop_last=False,
        train_split=data_params.get('train_split', 'train800'), **common)
    _, test_loader, _ = make_dataset(
        transform=test_transform, training=False, drop_last=False,
        test_split=data_params.get('test_split', 'test'), **common)
    val_loader = None
    if data_params.get('val_split') is not None:
        _, val_loader, _ = make_dataset(
            transform=test_transform, training=False, drop_last=False,
            test_split=data_params['val_split'], **common)
    ipe = len(train_loader)
    logger.info(f"#train: {len(train_loader.dataset)} (ipe {ipe}), #test: {len(test_loader.dataset)}")

    # -- optimization: AdamW + cosine, as specified in the paper
    opt_params = params['opt']
    num_epochs = opt_params['epochs']
    optimizer = get_finetune_optimizer(
        model,
        optimizer_name=opt_params.get('name', 'adamw'),
        base_lr=opt_params['lr']['base_lr'],
        weight_decay=opt_params['weight_decay'],
        bias_decay=opt_params.get('bias_decay', False),
        norm_decay=opt_params.get('norm_decay', False),
        world_size=world_size,
        batch_size_per_replica=data_params['batch_size_per_replica'],
        base_lr_batch_size=opt_params['lr'].get('base_lr_batch_size', 256),
        auto_lr_scaling=opt_params['lr'].get('auto_lr_scaling', False),
    )
    lr_scheduler = get_warmup_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=int(opt_params['lr'].get('warmup_epochs', 0) * ipe),
        start_lr=opt_params['lr'].get('start_lr', opt_params['lr']['base_lr']),
        ref_lr=opt_params['lr']['base_lr'],
        T_max=num_epochs * ipe,
        final_lr=opt_params['lr'].get('final_lr', 0.0),
    )
    use_bfloat16 = opt_params.get('use_bfloat16', False)
    scaler = get_gradient_scaler(use_bf16=use_bfloat16, device=device)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                        find_unused_parameters=False)

    start_epoch = 1
    resume_path = params.get('resume', {}).get('resume_path', None)
    if resume_path is not None:
        assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!"
        model, optimizer, scaler, lr_scheduler, start_epoch, _ = load_downstream_checkpoint(
            resume_path, model, optimizer, scaler, lr_scheduler)
        start_epoch += 1

    eval_freq = params['logging']['eval']['eval_epoch_freq']
    best_val, best_test, final_test = -1.0, -1.0, -1.0
    model.train()
    logger.info(f"Starting Clevr/{task} linear evaluation for {num_epochs} epochs")

    for epoch in range(start_epoch, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_bfloat16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if opt_params.get('clip_grad', 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt_params['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if opt_params.get('clip_grad', 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt_params['clip_grad'])
                optimizer.step()
            lr_scheduler.step()

            loss_meter.step(loss.item())
            acc_meter.step((logits.argmax(dim=1) == labels).float().mean().item() * 100.0)
            if step % params['logging']['log_step_freq'] == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step}/{ipe}] "
                            f"Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.2f}% "
                            f"LR: {lr_scheduler.get_current_lr():.2e}")
                loggers.log({'Loss/train': loss_meter.avg, 'Acc/train': acc_meter.avg,
                             'LR/lr': lr_scheduler.get_current_lr()}, step=(epoch - 1) * ipe + step)

        if epoch % eval_freq == 0 or epoch == num_epochs:
            test_acc = evaluate(model, test_loader, device, use_bfloat16)
            metrics = {'Acc/test': test_acc}
            if val_loader is not None:
                val_acc = evaluate(model, val_loader, device, use_bfloat16)
                metrics['Acc/val'] = val_acc
                if val_acc > best_val:
                    best_val, best_test = val_acc, test_acc
            final_test = test_acc
            logger.info(f"[epoch {epoch}] " + ", ".join(f"{k}: {v:.2f}%" for k, v in metrics.items()))
            loggers.log(metrics, step=epoch * ipe)

            if is_main() and params['logging']['ckpt'].get('save_latest', True):
                save_downstream_checkpoint(
                    save_path=os.path.join(ckpt_dir, f'clevr_{task}_latest.pth'),
                    step=epoch * ipe, epoch=epoch, model=model, opt=optimizer,
                    scaler=scaler, lr_scheduler=lr_scheduler, best_metric=best_test)

    logger.info(f"Clevr/{task} final test accuracy: {final_test:.2f}%"
                + (f" | val-selected test accuracy: {best_test:.2f}% (val {best_val:.2f}%)"
                   if val_loader is not None else ""))
    if is_main():
        import yaml
        with open(os.path.join(run['log_dir'], 'eval_results.yaml'), 'w') as f:
            yaml.safe_dump({'task': f'clevr_{task}', 'test_acc': float(final_test),
                            'val_selected_test_acc': float(best_test),
                            'best_val_acc': float(best_val)}, f)
    loggers.close()
    teardown()
