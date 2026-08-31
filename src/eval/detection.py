"""MS-COCO object detection / instance segmentation transfer (Mask R-CNN).

Protocol (paper, Sec. "Downstream Evaluation Details"):
a Mask R-CNN detector with an FPN is built on the pre-trained ViT-Base encoder
(feature maps of blocks 3, 5, 7, 11) and fine-tuned end-to-end on COCO 2017 for
25 epochs with AdamW (lr 1e-4, weight decay 0.1), linear warmup during the first
epoch followed by cosine decay, a global batch size of 16 and stochastic depth
with a maximum drop-path rate of 0.1.  AP^box and AP^mask are reported on
val2017.
"""

import os

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from src.dataset import make_dataset
from src.dataset.coco import COCO_NUM_CLASSES, collate_detection
from src.eval.common import setup_experiment, build_pretrained_encoder, teardown, \
    ShardSampler, gather_object
from src.models import init_detection_model
from src.utils.distributed import is_main_process as is_main
from src.utils.log import AverageMeter
from src.utils.opt.optimzer import get_finetune_optimizer, save_downstream_checkpoint, \
    load_downstream_checkpoint
from src.utils.opt.scaler import get_gradient_scaler
from src.utils.opt.scheduler import get_warmup_cosine_lr_scheduler


def encode_masks(masks: torch.Tensor, threshold: float = 0.5):
    """(N, 1, H, W) soft masks -> COCO RLEs."""
    from pycocotools import mask as coco_mask

    rles = []
    for m in masks:
        binary = np.asfortranarray((m[0].numpy() > threshold).astype(np.uint8))
        rle = coco_mask.encode(binary)
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles


@torch.no_grad()
def evaluate(model, loader, device, use_bfloat16=False, max_dets=100):
    """Run the detector over val2017 and score it with COCOeval."""
    model.eval()
    results = []
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        with torch.amp.autocast("cuda", enabled=use_bfloat16):
            outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].float().cpu()
            scores = output["scores"].float().cpu()
            labels = output["labels"].cpu()
            keep = scores.argsort(descending=True)[:max_dets]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            rles = encode_masks(output["masks"].float().cpu()[keep])
            xywh = boxes.clone()
            xywh[:, 2:] -= xywh[:, :2]
            for box, score, label, rle in zip(xywh.tolist(), scores.tolist(), labels.tolist(), rles):
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [round(c, 3) for c in box],
                    "score": float(score),
                    "segmentation": rle,
                })
    model.train()

    gathered = [r for shard in gather_object(results) for r in shard]
    metrics = {"AP_box": 0.0, "AP_mask": 0.0}
    if is_main() and len(gathered) > 0:
        import copy

        from pycocotools.cocoeval import COCOeval

        # COCOeval rewrites the ground-truth polygons in place (annToRLE), so it
        # is given a copy -- the dataset keeps returning usable segmentations.
        coco_gt = copy.deepcopy(loader.dataset.coco)
        coco_dt = coco_gt.loadRes(gathered)
        for iou_type, key in (("bbox", "box"), ("segm", "mask")):
            coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics[f"AP_{key}"] = 100.0 * float(coco_eval.stats[0])
            metrics[f"AP50_{key}"] = 100.0 * float(coco_eval.stats[1])
            metrics[f"AP75_{key}"] = 100.0 * float(coco_eval.stats[2])
    metrics = gather_object(metrics)[0]
    return metrics


def main(params, args):
    run = setup_experiment(params, args)
    device, rank, world_size = run['device'], run['rank'], run['world_size']
    logger, loggers, ckpt_dir = run['logger'], run['loggers'], run['ckpt_dir']

    data_params, opt_params = params['data'], params['opt']
    det_params = params.get('detection', {})
    num_classes = data_params.get('num_classes', COCO_NUM_CLASSES)
    img_size = det_params.get('img_size', 1024)

    # -- model: pre-trained ViT + FPN + Mask R-CNN heads.
    #    The encoder keeps its pre-training grid; positional embeddings are
    #    interpolated to the detection resolution at every forward.
    encoder = build_pretrained_encoder(params, device)
    model = init_detection_model(
        encoder,
        num_classes=num_classes,
        out_layers=params['model'].get('out_layers', [3, 5, 7, 11]),
        fpn_channels=det_params.get('fpn_channels', 256),
        img_size=img_size,
        window_size=params['model'].get('window_size', 14),
        use_checkpoint=params['model'].get('use_checkpoint', True),
        freeze_backbone=params['model'].get('freeze_backbone', False),
    ).to(device)

    # -- data
    common = dict(
        dataset_name=data_params['dataset_name'],
        pin_mem=data_params['pin_memory'],
        num_workers=data_params['num_workers'],
        world_size=world_size,
        rank=rank,
        root_path=data_params['root_path'],
        collator=collate_detection,
    )
    _, train_loader, train_sampler = make_dataset(
        batch_size=data_params['batch_size_per_replica'], training=True, drop_last=True, **common)
    val_dataset, _, _ = make_dataset(
        batch_size=1, training=False, drop_last=False, **common)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_params.get('eval_batch_size_per_replica', 1),
        sampler=ShardSampler(val_dataset, world_size, rank),
        num_workers=data_params['num_workers'], pin_memory=data_params['pin_memory'],
        collate_fn=collate_detection)
    ipe = len(train_loader)
    logger.info(f"#train: {len(train_loader.dataset)} (ipe {ipe}), #val: {len(val_dataset)}")

    # -- optimization: AdamW, 1 epoch linear warmup then cosine
    num_epochs = opt_params['epochs']
    optimizer = get_finetune_optimizer(
        model,
        optimizer_name=opt_params.get('name', 'adamw'),
        base_lr=opt_params['lr']['base_lr'],
        weight_decay=opt_params['weight_decay'],
        bias_decay=opt_params.get('bias_decay', False),
        norm_decay=opt_params.get('norm_decay', False),
        backbone_lr_scale=opt_params['lr'].get('backbone_lr_scale', 1.0),
        backbone_prefixes=('body', 'backbone'),
        world_size=world_size,
        batch_size_per_replica=data_params['batch_size_per_replica'],
        base_lr_batch_size=opt_params['lr'].get('base_lr_batch_size', 16),
        auto_lr_scaling=opt_params['lr'].get('auto_lr_scaling', False),
    )
    lr_scheduler = get_warmup_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=int(opt_params['lr'].get('warmup_epochs', 1) * ipe),
        start_lr=opt_params['lr'].get('start_lr', 0.0),
        ref_lr=opt_params['lr']['base_lr'],
        T_max=num_epochs * ipe,
        final_lr=opt_params['lr'].get('final_lr', 0.0),
    )
    use_bfloat16 = opt_params.get('use_bfloat16', True)
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
    log_freq = params['logging']['log_step_freq']
    best = {'AP_box': -1.0, 'AP_mask': -1.0}
    model.train()
    logger.info(f"Starting COCO fine-tuning for {num_epochs} epochs "
                f"(global batch {data_params['batch_size_per_replica'] * world_size})")

    for epoch in range(start_epoch, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        meters = {}
        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=use_bfloat16):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

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

            for k, v in list(loss_dict.items()) + [('loss', loss)]:
                meters.setdefault(k, AverageMeter()).step(float(v))
            if step % log_freq == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step}/{ipe}] "
                            + " ".join(f"{k}: {m.avg:.4f}" for k, m in meters.items())
                            + f" lr: {lr_scheduler.get_current_lr():.2e}")
                loggers.log({f'Loss/{k}': m.avg for k, m in meters.items()}
                            | {'LR/lr': lr_scheduler.get_current_lr()},
                            step=(epoch - 1) * ipe + step)

        if epoch % eval_freq == 0 or epoch == num_epochs:
            results = evaluate(model, val_loader, device, use_bfloat16,
                               max_dets=params['logging']['eval'].get('max_dets', 100))
            logger.info(f"[epoch {epoch}] " + ", ".join(f"{k}: {v:.2f}" for k, v in results.items()))
            loggers.log({f'Val/{k}': v for k, v in results.items()}, step=epoch * ipe)
            if results.get('AP_box', -1) > best['AP_box']:
                best = results
                if is_main() and params['logging']['ckpt'].get('save_best', True):
                    save_downstream_checkpoint(
                        os.path.join(ckpt_dir, 'coco_maskrcnn_best.pth'), epoch * ipe, epoch,
                        model, optimizer, scaler, lr_scheduler, best_metric=best['AP_box'])

        if is_main() and params['logging']['ckpt'].get('save_latest', True):
            save_downstream_checkpoint(
                os.path.join(ckpt_dir, 'coco_maskrcnn_latest.pth'), epoch * ipe, epoch,
                model, optimizer, scaler, lr_scheduler, best_metric=best['AP_box'])

    logger.info(f"COCO best AP^box: {best['AP_box']:.2f}, AP^mask: {best['AP_mask']:.2f}")
    if is_main():
        import yaml
        with open(os.path.join(run['log_dir'], 'eval_results.yaml'), 'w') as f:
            yaml.safe_dump({'task': 'coco_detection_instance_segmentation',
                            **{k: float(v) for k, v in best.items()}}, f)
    loggers.close()
    teardown()
