"""Shared plumbing for the downstream evaluations (probing / CLEVR / COCO / ADE20K).

Keeps the per-task scripts focused on their protocol: this module owns the
distributed / logging / checkpoint-directory boilerplate and the construction of
the pre-trained encoder that every downstream task starts from.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist

from src.models import init_target_encoder
from src.utils.distributed import init_distributed_mode, get_rank, get_world_size
from src.utils.distributed import is_main_process as is_main
from src.utils.log import setup_logging, get_logger
from src.utils.opt.optimzer import load_jepa_target_encoder_weights


class RunLoggers:
    """Thin fan-out over the CSV / TensorBoard / W&B loggers (rank-0 only)."""

    def __init__(self, params, rank):
        self.enabled = is_main()
        self.csv = self.tb = self.wandb = None
        if not self.enabled:
            return
        log_dir = params['logging']['log_dir']
        if params['logging'].get('use_csv', False):
            from src.utils.log import CSVLogger
            self.csv = CSVLogger(log_dir=log_dir, rank=rank)
        if params['logging'].get('use_tensorboard', False):
            from src.utils.log import TensorboardLogger
            self.tb = TensorboardLogger(log_dir=os.path.join(log_dir, 'tb_logs'), rank=rank)
        if params['logging'].get('wandb', {}).get('use_wandb', False):
            from src.utils.log import WandbLogger
            wb = params['logging']['wandb']
            self.wandb = WandbLogger(
                project_name=wb['project_name'],
                run_name=wb['run_name'],
                entity=wb['entity'],
                config=params,
                rank=rank,
            )

    def log(self, metrics: dict, step: int):
        if not self.enabled:
            return
        if self.csv is not None:
            self.csv.log_metrics({**metrics, 'step': step}, step=step)
        if self.tb is not None:
            self.tb.log_metrics(metrics, step=step)
        if self.wandb is not None:
            self.wandb.log_metrics(metrics)

    def close(self):
        if not self.enabled:
            return
        for lg in (self.csv, self.tb, self.wandb):
            if lg is not None:
                lg.close()


def setup_experiment(params, args):
    """Initialize distributed mode, logging and the output directories."""
    init_distributed_mode(args)
    rank, world_size = get_rank(), get_world_size()
    device = torch.device(f'cuda:{args.gpu}')

    setup_logging(rank, world_size)
    logger = get_logger()
    logger.info(f"Using device: {device}, rank: {rank}, world_size: {world_size}")

    log_dir = params['logging']['log_dir']
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if is_main():
        for d in (log_dir, ckpt_dir, os.path.join(log_dir, 'tb_logs'),
                  os.path.join(log_dir, 'visualizations')):
            os.makedirs(d, exist_ok=True)
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            import yaml
            yaml.safe_dump(params, f, sort_keys=False)
    if dist.is_initialized():
        dist.barrier()

    loggers = RunLoggers(params, rank)
    return dict(device=device, rank=rank, world_size=world_size, logger=logger,
                log_dir=log_dir, ckpt_dir=ckpt_dir, loggers=loggers)


def build_pretrained_encoder(params, device, img_size: Optional[int] = None):
    """Instantiate the ViT encoder and load the pre-trained (target-encoder) weights."""
    model_params = params['model']
    encoder = init_target_encoder(
        device,
        patch_size=model_params['patch_size'],
        model_name=model_params['model_name'],
        crop_size=img_size if img_size is not None else model_params.get('crop_size', 224),
        use_masked_vit=model_params.get('use_masked_vit', True),
        use_class_token=model_params.get('use_class_token', True),
        drop_path_rate=model_params.get('drop_path_rate', 0.0),
    )
    pretrained_weights = model_params['pretrained_weights']
    assert pretrained_weights is not None, \
        "Please provide `model.pretrained_weights` (a JEPA pre-training checkpoint)."
    encoder = load_jepa_target_encoder_weights(
        encoder, pretrained_weights, device,
        strict=model_params.get('strict_load', True),
    )
    return encoder


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_object(obj):
    """Gather arbitrary picklable objects from every rank onto every rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None for _ in range(get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def teardown():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class ShardSampler(torch.utils.data.Sampler):
    """Evaluation sampler that shards a dataset across ranks *without* padding.

    `DistributedSampler` repeats a few samples so every rank sees the same
    number of batches; for metric accumulation (mIoU, COCO AP) those duplicates
    would be counted twice, so evaluation uses this exact, non-padding shard.
    """

    def __init__(self, dataset, num_replicas: int = 1, rank: int = 0):
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def infinite_loader(loader, sampler=None):
    """Yield batches forever, re-shuffling the distributed sampler every pass."""
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
