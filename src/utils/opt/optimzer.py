import torch
import torch.optim as optim
from torch import nn
from itertools import chain

from src.utils.opt.scheduler import LinearEMASchedule,WarmupCosineLRSchedule,CosineWDSchedule, MultiStepValuesLRSchedule
from src.utils.log import get_logger
logger = get_logger(__name__)

def _compute_scaled_lr(base_lr, world_size, batch_size_per_replica, base_lr_batch_size, auto_lr_scaling=False):
    base_lr = base_lr
    if auto_lr_scaling:
        per_device_bs = batch_size_per_replica
        accum = 1
        effective_bs = per_device_bs * world_size * accum
        scale = effective_bs / base_lr_batch_size
        return scale * base_lr
    else:
        return base_lr

def get_wd_filter(
    bias_decay: bool = False,
    norm_decay: bool = False,
):
    """
    Create a weight decay filter function based on the provided conditions.
    params:
        bias_decay (bool): Whether to apply weight decay to bias parameters.
        norm_decay (bool): Whether to apply weight decay to normalization layers.
    returns:
        function: A function that determines if weight decay should be applied to a parameter.
    """
    def is_bias(n: str) -> bool:
        return n.endswith("bias")

    def is_norm_like(n: str, p) -> bool:
        return len(getattr(p, "shape", ())) == 1

    def should_decay(n, p) -> bool:
        if is_bias(n):
            return bias_decay
        if is_norm_like(n, p):
            return norm_decay
        return True
    return should_decay

def get_optimizer(
    optimizer_name: str,
    param_groups = None,
    **kwargs
):
    """
    Create an optimizer based on the provided name.
    Args:
        optimizer_name (str): Name of the optimizer ('sgd' or 'adamw').
    Returns:
        torch.optim.Optimizer: The corresponding optimizer class.
    """
    assert param_groups is not None, "param_groups must be provided"
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(param_groups, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(param_groups, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_pretrain_optimizer(
    encoder: nn.Module,
    predictor: nn.Module,
    optimizer_name: str = 'adamw',
    bias_decay: bool = False,
    norm_decay: bool = False,   
    mlp_projector: nn.Module = None,
):
    """
    Create optimizer for pretraining stage.

    Args:
        encoder (nn.Module): The encoder model.
        predictor (nn.Module): The predictor model.
        optimizer_name (str): Name of optimizer ('adamw', 'sgd', etc.)
        bias_decay (bool): Apply weight decay to bias params if True.
        norm_decay (bool): Apply weight decay to norm layers if True.
        mlp_projector (nn.Module): Whether to include MLP projector parameters in the optimizer.
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    wd_filter = get_wd_filter(bias_decay, norm_decay)

    # Materialize once to avoid exhausting generators multiple times
    all_named = list(chain(encoder.named_parameters(), predictor.named_parameters()))
    if mlp_projector is not None:
        all_named += list(mlp_projector.named_parameters())
    decay_params    = [p for n, p in all_named if wd_filter(n, p)]
    no_decay_params = [p for n, p in all_named if not wd_filter(n, p)]

    param_groups = [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0, "WD_exclude": True},
    ]

    optimizer = get_optimizer(optimizer_name, param_groups)
    return optimizer

def get_probing_optimizer(
    model: nn.Module,
    optimizer_name: str = 'sgd',
    bias_decay: bool = False,
    norm_decay: bool = False,
    weight_decay: float = 0.0,
    base_lr: float = 0.01,
    world_size: int = 8,
    batch_size_per_replica: int = 32,
    base_lr_batch_size: int = 256,
    auto_lr_scaling: bool = False,
    **kwargs,
):
    """
    Create optimizer for linear probing stage.

    Args:
        model (nn.Module): The linear evaluation model.
        optimizer_name (str): Name of optimizer ('adamw', 'sgd', etc.)
        bias_decay (bool): Apply weight decay to bias params if True.
        norm_decay (bool): Apply weight decay to norm layers if True.
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    wd_filter = get_wd_filter(bias_decay, norm_decay)

    all_named = list(model.named_parameters())
    decay_params    = [p for n, p in all_named if wd_filter(n, p)]
    no_decay_params = [p for n, p in all_named if not wd_filter(n, p)]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0, "WD_exclude": True},
    ]
    lr = _compute_scaled_lr(
        base_lr,
        world_size,
        batch_size_per_replica,
        base_lr_batch_size,
        auto_lr_scaling,
    )
    optimizer = get_optimizer(optimizer_name, param_groups, lr=lr, **kwargs)
    return optimizer

def save_jepa_checkpoint(
    save_path: str,
    epoch: int, 
    encoder: nn.Module,
    predictor: nn.Module,
    target_encoder: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    lr_scheduler: WarmupCosineLRSchedule,
    wd_scheduler: CosineWDSchedule,
    ema_scheduler: LinearEMASchedule,
):
    checkpoint = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'wd_scheduler': wd_scheduler.state_dict(),
        'ema_scheduler': ema_scheduler.state_dict(),
    }
    if target_encoder is not None:
        checkpoint['target_encoder'] = target_encoder.state_dict()
    torch.save(checkpoint, save_path)
    logger.info(f'Saved checkpoint at epoch {epoch} to {save_path}')

def load_checkpoint(
    resume_path: str,
    encoder: nn.Module,
    predictor: nn.Module,
    target_encoder: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    lr_scheduler: WarmupCosineLRSchedule,
    wd_scheduler: CosineWDSchedule,
    ema_scheduler: LinearEMASchedule,
):
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=False)
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading predictor
    pretrained_dict = checkpoint['predictor']
    msg = predictor.load_state_dict(pretrained_dict)
    logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

    # -- loading target_encoder
    if target_encoder is not None:
        logger.info(list(checkpoint.keys()))
        pretrained_dict = checkpoint['target_encoder']
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')

    # -- loading optimizers
    opt.load_state_dict(checkpoint['opt'])
    scaler.load_state_dict(checkpoint['scaler'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    wd_scheduler.load_state_dict(checkpoint['wd_scheduler'])
    ema_scheduler.load_state_dict(checkpoint['ema_scheduler'])

    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {resume_path}')
    del checkpoint
    return encoder, predictor, target_encoder, opt, scaler, epoch

def load_jepa_target_encoder_weights(
    target_encoder: nn.Module,
    checkpoint_path: str,
    map_device: str = 'cpu',
    strict: bool = True,
):  
    checkpoint = torch.load(checkpoint_path, map_location=map_device, weights_only=False)
    target_encoder_ckpt = checkpoint['target_encoder'] if 'target_encoder' in checkpoint else checkpoint['encoder']
    
    if "module." in list(target_encoder_ckpt.keys())[0]:
        # Remove 'module.' prefix if present (from DDP training)
        target_encoder_ckpt = {k.replace("module.", ""): v for k, v in target_encoder_ckpt.items()}
    msg = target_encoder.load_state_dict(target_encoder_ckpt, strict=strict)
    logger.info(f'loaded target encoder from {checkpoint_path} with msg: {msg}')
    return target_encoder

def load_jepa_all_weights_for_analysis(
    encoder: nn.Module,
    predictor: nn.Module,
    target_encoder: nn.Module,
    checkpoint_path: str,
    map_device: str = 'cpu',
):
    checkpoint = torch.load(checkpoint_path, map_location=map_device, weights_only=False)
    
    encoder_ckpt = checkpoint['encoder']
    predictor_ckpt = checkpoint['predictor']
    target_encoder_ckpt = checkpoint['target_encoder'] if 'target_encoder' in checkpoint else checkpoint['encoder']
    
    if "module." in list(encoder_ckpt.keys())[0]:
        # Remove 'module.' prefix if present (from DDP training)
        encoder_ckpt = {k.replace("module.", ""): v for k, v in encoder_ckpt.items()}
    if "module." in list(predictor_ckpt.keys())[0]:
        predictor_ckpt = {k.replace("module.", ""): v for k, v in predictor_ckpt.items()}
    if "module." in list(target_encoder_ckpt.keys())[0]:
        target_encoder_ckpt = {k.replace("module.", ""): v for k, v in target_encoder_ckpt.items()}
    
    encoder.load_state_dict(encoder_ckpt)
    predictor.load_state_dict(predictor_ckpt)
    target_encoder.load_state_dict(target_encoder_ckpt)
    
    return encoder, predictor, target_encoder
    
def save_probing_checkpoint(
    save_path: str,
    epoch: int, 
    model_with_head: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    lr_scheduler: MultiStepValuesLRSchedule,
):
    checkpoint = {
        'epoch': epoch,
        'model': model_with_head.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)
    logger.info(f'Saved probing checkpoint at epoch {epoch} to {save_path}')

def load_probing_checkpoint(
    resume_path: str,
    model_with_head: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    lr_scheduler: MultiStepValuesLRSchedule,
):
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=False)
    epoch = checkpoint['epoch']

    # -- loading model
    pretrained_dict = checkpoint['model']
    msg = model_with_head.load_state_dict(pretrained_dict)
    logger.info(f'loaded probing model from epoch {epoch} with msg: {msg}')

    # -- loading optimizers
    opt.load_state_dict(checkpoint['opt'])
    scaler.load_state_dict(checkpoint['scaler'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    logger.info(f'loaded optimizers from epoch {epoch}')
    logger.info(f'read-path: {resume_path}')
    del checkpoint
    return model_with_head, opt, scaler, lr_scheduler, epoch
    
    
    

def get_finetune_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    base_lr: float = 1e-4,
    weight_decay: float = 0.05,
    bias_decay: bool = False,
    norm_decay: bool = False,
    backbone_lr_scale: float = 1.0,
    backbone_prefixes: tuple = ('backbone', 'body'),
    world_size: int = 8,
    batch_size_per_replica: int = 2,
    base_lr_batch_size: int = 16,
    auto_lr_scaling: bool = False,
    **kwargs,
):
    """Create the optimizer used by the dense downstream tasks (COCO / ADE20K).

    Parameters are split into (decay / no-decay) x (backbone / head) groups, so
    the pre-trained encoder can be given a smaller learning rate through
    `backbone_lr_scale` while everything is driven by a single schedule.
    """
    wd_filter = get_wd_filter(bias_decay, norm_decay)

    groups = {
        ('backbone', True): {"params": [], "weight_decay": weight_decay, "lr_scale": backbone_lr_scale},
        ('backbone', False): {"params": [], "weight_decay": 0.0, "WD_exclude": True, "lr_scale": backbone_lr_scale},
        ('head', True): {"params": [], "weight_decay": weight_decay, "lr_scale": 1.0},
        ('head', False): {"params": [], "weight_decay": 0.0, "WD_exclude": True, "lr_scale": 1.0},
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = 'backbone' if any(n.startswith(pre + '.') or f'.{pre}.' in n for pre in backbone_prefixes) else 'head'
        groups[(where, wd_filter(n, p))]["params"].append(p)

    param_groups = [g for g in groups.values() if len(g["params"]) > 0]
    lr = _compute_scaled_lr(
        base_lr, world_size, batch_size_per_replica, base_lr_batch_size, auto_lr_scaling)
    for g in param_groups:
        g["lr"] = lr * g["lr_scale"]
    optimizer = get_optimizer(optimizer_name, param_groups, lr=lr, **kwargs)
    logger.info(f'Built {optimizer_name} with {len(param_groups)} param groups, lr={lr:.2e}, '
                f'backbone_lr_scale={backbone_lr_scale}')
    return optimizer


def save_downstream_checkpoint(
    save_path: str,
    step: int,
    epoch: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler=None,
    lr_scheduler=None,
    best_metric=None,
):
    """Checkpoint for the detection / segmentation / low-level transfer tasks."""
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'best_metric': best_metric,
    }
    torch.save(checkpoint, save_path)
    logger.info(f'Saved downstream checkpoint (epoch {epoch}, step {step}) to {save_path}')


def load_downstream_checkpoint(
    resume_path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer = None,
    scaler=None,
    lr_scheduler=None,
):
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=False)
    msg = model.load_state_dict(checkpoint['model'])
    logger.info(f"loaded downstream model from epoch {checkpoint.get('epoch')} with msg: {msg}")
    if opt is not None and checkpoint.get('opt') is not None:
        opt.load_state_dict(checkpoint['opt'])
    if scaler is not None and checkpoint.get('scaler') is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    if lr_scheduler is not None and checkpoint.get('lr_scheduler') is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return model, opt, scaler, lr_scheduler, checkpoint.get('epoch', 0), checkpoint.get('step', 0)
