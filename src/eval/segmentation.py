"""ADE20K semantic segmentation transfer (UPerNet on the pre-trained ViT).

Protocol (paper, Sec. "Downstream Evaluation Details"):
a UPerNet-style decoder is attached to the ViT-Base backbone, taking the feature
maps of blocks 3, 5, 7 and 11; the model is fine-tuned with a 512x512 input crop
using AdamW (lr 1e-4, weight decay 0.05) and evaluated with sliding-window
inference (512x512 crop, stride 341x341), reporting mIoU on the validation set.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.dataset import make_dataset
from src.dataset.ade20k import ADE20K_NUM_CLASSES, IGNORE_INDEX, collate_segmentation
from src.eval.common import setup_experiment, build_pretrained_encoder, all_reduce_sum, \
    teardown, ShardSampler, infinite_loader
from src.models import init_segmentation_model
from src.utils.distributed import is_main_process as is_main
from src.utils.log import AverageMeter
from src.utils.opt.optimzer import get_finetune_optimizer, save_downstream_checkpoint, \
    load_downstream_checkpoint
from src.utils.opt.scaler import get_gradient_scaler
from src.utils.opt.scheduler import get_poly_lr_scheduler


@torch.no_grad()
def slide_inference(model, img, num_classes, crop_size=512, stride=341, use_bfloat16=False):
    """Sliding-window inference over a full-resolution image (1, 3, H, W)."""
    _, _, h_img, w_img = img.shape
    h_grids = max(h_img - crop_size + stride - 1, 0) // stride + 1
    w_grids = max(w_img - crop_size + stride - 1, 0) // stride + 1
    logits = img.new_zeros((1, num_classes, h_img, w_img), dtype=torch.float32)
    count = img.new_zeros((1, 1, h_img, w_img), dtype=torch.float32)

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1, x1 = h_idx * stride, w_idx * stride
            y2, x2 = min(y1 + crop_size, h_img), min(x1 + crop_size, w_img)
            y1, x1 = max(y2 - crop_size, 0), max(x2 - crop_size, 0)
            crop = img[:, :, y1:y2, x1:x2]
            # -- pad crops smaller than the window (small images / borders)
            pad_h, pad_w = crop_size - crop.shape[2], crop_size - crop.shape[3]
            if pad_h > 0 or pad_w > 0:
                crop = F.pad(crop, (0, max(0, pad_w), 0, max(0, pad_h)))
            with torch.amp.autocast("cuda", enabled=use_bfloat16):
                crop_logits = model(crop)
            crop_logits = crop_logits.float()[:, :, :y2 - y1, :x2 - x1]
            logits[:, :, y1:y2, x1:x2] += crop_logits
            count[:, :, y1:y2, x1:x2] += 1
    assert (count == 0).sum() == 0, "sliding window did not cover the whole image"
    return logits / count


def confusion_update(conf, pred, label, num_classes):
    """Accumulate a (num_classes, num_classes) confusion matrix, ignoring 255."""
    valid = label != IGNORE_INDEX
    idx = label[valid].to(torch.int64) * num_classes + pred[valid].to(torch.int64)
    conf += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf


def metrics_from_confusion(conf):
    conf = conf.double()
    inter = conf.diag()
    gt = conf.sum(dim=1)
    pred = conf.sum(dim=0)
    union = gt + pred - inter
    iou = inter / union.clamp(min=1)
    acc = inter / gt.clamp(min=1)
    valid = gt > 0
    return {
        'mIoU': 100.0 * iou[valid].mean().item(),
        'mAcc': 100.0 * acc[valid].mean().item(),
        'aAcc': 100.0 * (inter.sum() / gt.sum().clamp(min=1)).item(),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes, crop_size=512, stride=341, use_bfloat16=False):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    for images, labels in loader:
        # -- one image at a time (validation images keep their original size)
        img = images[0].unsqueeze(0).to(device, non_blocking=True)
        label = labels[0].to(device, non_blocking=True)
        logits = slide_inference(model, img, num_classes, crop_size, stride, use_bfloat16)
        pred = logits.argmax(dim=1)[0]
        conf = confusion_update(conf, pred.flatten(), label.flatten(), num_classes)
    all_reduce_sum(conf)
    model.train()
    return metrics_from_confusion(conf.cpu())


def main(params, args):
    run = setup_experiment(params, args)
    device, rank, world_size = run['device'], run['rank'], run['world_size']
    logger, loggers, ckpt_dir = run['logger'], run['loggers'], run['ckpt_dir']

    data_params, opt_params = params['data'], params['opt']
    seg_params = params.get('segmentation', {})
    num_classes = data_params.get('num_classes', ADE20K_NUM_CLASSES)
    crop_size = data_params.get('crop_size', 512)

    # -- model: pre-trained ViT + UPerNet decoder over blocks 3/5/7/11.
    #    The encoder keeps its pre-training grid; positional embeddings are
    #    interpolated to the 512x512 crop at every forward (see ViTDenseBackbone).
    encoder = build_pretrained_encoder(params, device)
    model = init_segmentation_model(
        encoder,
        num_classes=num_classes,
        out_layers=params['model'].get('out_layers', [3, 5, 7, 11]),
        channels=seg_params.get('channels', 512),
        pool_scales=seg_params.get('pool_scales', [1, 2, 3, 6]),
        dropout=seg_params.get('dropout', 0.1),
        use_aux_head=seg_params.get('use_aux_head', True),
        aux_index=seg_params.get('aux_index', 2),
        aux_channels=seg_params.get('aux_channels', 256),
        window_size=params['model'].get('window_size', 0),
        use_checkpoint=params['model'].get('use_checkpoint', False),
        freeze_backbone=params['model'].get('freeze_backbone', False),
    ).to(device)
    aux_weight = seg_params.get('aux_loss_weight', 0.4)

    # -- data
    common = dict(
        dataset_name=data_params['dataset_name'],
        pin_mem=data_params['pin_memory'],
        num_workers=data_params['num_workers'],
        world_size=world_size,
        rank=rank,
        root_path=data_params['root_path'],
        crop_size=crop_size,
        base_size=data_params.get('base_size', (2048, 512)),
        ratio_range=data_params.get('ratio_range', (0.5, 2.0)),
        test_scale=data_params.get('test_scale', (2048, 512)),
    )
    _, train_loader, train_sampler = make_dataset(
        batch_size=data_params['batch_size_per_replica'], training=True, drop_last=True, **common)
    val_dataset, _, _ = make_dataset(
        batch_size=1, training=False, drop_last=False, **common)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=ShardSampler(val_dataset, world_size, rank),
        num_workers=data_params['num_workers'], pin_memory=data_params['pin_memory'],
        collate_fn=collate_segmentation)
    logger.info(f"#train: {len(train_loader.dataset)}, #val: {len(val_dataset)} "
                f"({len(val_loader)} on this rank)")

    # -- optimization: AdamW + poly schedule (iteration based, as usual for ADE20K)
    total_iters = opt_params['iterations']
    optimizer = get_finetune_optimizer(
        model,
        optimizer_name=opt_params.get('name', 'adamw'),
        base_lr=opt_params['lr']['base_lr'],
        weight_decay=opt_params['weight_decay'],
        bias_decay=opt_params.get('bias_decay', False),
        norm_decay=opt_params.get('norm_decay', False),
        backbone_lr_scale=opt_params['lr'].get('backbone_lr_scale', 1.0),
        world_size=world_size,
        batch_size_per_replica=data_params['batch_size_per_replica'],
        base_lr_batch_size=opt_params['lr'].get('base_lr_batch_size', 16),
        auto_lr_scaling=opt_params['lr'].get('auto_lr_scaling', False),
    )
    lr_scheduler = get_poly_lr_scheduler(
        optimizer=optimizer,
        base_lr=opt_params['lr']['base_lr'],
        total_steps=total_iters,
        warmup_steps=opt_params['lr'].get('warmup_iters', 1500),
        warmup_ratio=opt_params['lr'].get('warmup_ratio', 1e-6),
        power=opt_params['lr'].get('power', 1.0),
        min_lr=opt_params['lr'].get('min_lr', 0.0),
    )
    use_bfloat16 = opt_params.get('use_bfloat16', True)
    scaler = get_gradient_scaler(use_bf16=use_bfloat16, device=device)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                        find_unused_parameters=False)

    start_iter = 0
    resume_path = params.get('resume', {}).get('resume_path', None)
    if resume_path is not None:
        assert os.path.isfile(resume_path), f"Resume path {resume_path} not found!"
        model, optimizer, scaler, lr_scheduler, _, start_iter = load_downstream_checkpoint(
            resume_path, model, optimizer, scaler, lr_scheduler)

    eval_freq = params['logging']['eval']['eval_iter_freq']
    ckpt_freq = params['logging']['ckpt'].get('ckpt_iter_freq', eval_freq)
    log_freq = params['logging']['log_step_freq']
    best = {'mIoU': -1.0}
    model.train()
    logger.info(f"Starting ADE20K fine-tuning for {total_iters} iterations "
                f"(global batch {data_params['batch_size_per_replica'] * world_size})")

    loss_meter, aux_meter = AverageMeter(), AverageMeter()
    batches = infinite_loader(train_loader, train_sampler)
    for it in range(start_iter, total_iters):
        images, labels = next(batches)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_bfloat16):
            logits, aux_logits = model(images)
            loss_main = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
            loss_aux = F.cross_entropy(aux_logits, labels, ignore_index=IGNORE_INDEX)
            loss = loss_main + aux_weight * loss_aux

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

        loss_meter.step(loss_main.item())
        aux_meter.step(loss_aux.item())
        if (it + 1) % log_freq == 0:
            logger.info(f"Iter [{it + 1}/{total_iters}] loss: {loss_meter.avg:.4f} "
                        f"aux: {aux_meter.avg:.4f} lr: {lr_scheduler.get_current_lr():.2e}")
            loggers.log({'Loss/decode': loss_meter.avg, 'Loss/aux': aux_meter.avg,
                         'LR/lr': lr_scheduler.get_current_lr()}, step=it + 1)
            loss_meter.reset(); aux_meter.reset()

        if (it + 1) % eval_freq == 0 or (it + 1) == total_iters:
            results = evaluate(
                model, val_loader, device, num_classes,
                crop_size=crop_size,
                stride=params['logging']['eval'].get('slide_stride', 341),
                use_bfloat16=use_bfloat16,
            )
            logger.info(f"[iter {it + 1}] " + ", ".join(f"{k}: {v:.2f}" for k, v in results.items()))
            loggers.log({f'Val/{k}': v for k, v in results.items()}, step=it + 1)
            if results['mIoU'] > best['mIoU']:
                best = results
                if is_main() and params['logging']['ckpt'].get('save_best', True):
                    save_downstream_checkpoint(
                        os.path.join(ckpt_dir, 'ade20k_upernet_best.pth'), it + 1, 0,
                        model, optimizer, scaler, lr_scheduler, best_metric=best['mIoU'])

        if is_main() and ((it + 1) % ckpt_freq == 0) and params['logging']['ckpt'].get('save_latest', True):
            save_downstream_checkpoint(
                os.path.join(ckpt_dir, 'ade20k_upernet_latest.pth'), it + 1, 0,
                model, optimizer, scaler, lr_scheduler, best_metric=best['mIoU'])

    logger.info(f"ADE20K best mIoU: {best['mIoU']:.2f} (mAcc {best['mAcc']:.2f}, aAcc {best['aAcc']:.2f})")
    if is_main():
        import yaml
        with open(os.path.join(run['log_dir'], 'eval_results.yaml'), 'w') as f:
            yaml.safe_dump({'task': 'ade20k_semantic_segmentation',
                            **{k: float(v) for k, v in best.items()}}, f)
    loggers.close()
    teardown()
