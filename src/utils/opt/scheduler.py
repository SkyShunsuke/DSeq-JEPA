import math
from bisect import bisect_right
import torch

def get_ema_scheduler(
    ema_start: float,
    ema_end: float,
    num_epochs: int,
    iters_per_epoch: int,
    ipe_scale: float = 1.0
) -> 'LinearEMASchedule':
    """
    Returns a LinearEMASchedule for EMA momentum scheduling.
    """
    return LinearEMASchedule(
        ema_start=ema_start,
        ema_end=ema_end,
        num_epochs=num_epochs,
        iters_per_epoch=iters_per_epoch,
        ipe_scale=ipe_scale
    )
    
def get_lambda_scheduler(
    lambda_start: float,
    lambda_end: float,
    num_epochs: int,
    iters_per_epoch: int,
    ipe_scale: float = 1.0
) -> 'LinearLambdaSchedule':
    """
    Returns a LinearLambdaSchedule for lambda scheduling.
    """
    return LinearLambdaSchedule(
        lambda_start=lambda_start,
        lambda_end=lambda_end,
        num_epochs=num_epochs,
        iters_per_epoch=iters_per_epoch,
        ipe_scale=ipe_scale
    )
def get_multi_step_values_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    milestones: list,
    values: list,
    iterations_per_epoch: int,
    start_step: int = 0,
) -> 'MultiStepValuesLRSchedule':
    """
    Returns a MultiStepValuesLRSchedule for learning rate scheduling.
    """
    return MultiStepValuesLRSchedule(
        optimizer=optimizer,
        milestones=milestones,
        values=values,
        iterations_per_epoch=iterations_per_epoch,
        start_step=start_step,
    )

def get_warmup_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    start_lr: float,
    ref_lr: float,
    T_max: int,
    final_lr: float = 0.,
    fix_lr_thres: int = -1,
    fix_strategy: str = 'const'
) -> 'WarmupCosineLRSchedule':
    """
    Returns a WarmupCosineLRSchedule for learning rate scheduling.
    """
    return WarmupCosineLRSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=ref_lr,
        T_max=T_max,
        final_lr=final_lr,
        fix_lr_thres=fix_lr_thres,
        fix_strategy=fix_strategy
    )

def get_cosine_wd_scheduler(
    optimizer: torch.optim.Optimizer,
    ref_wd: float,
    T_max: int,
    final_wd: float = 0.,
    fix_wd_thres: int = -1,
    fix_strategy: str = 'const'
) -> 'CosineWDSchedule':
    """
    Returns a CosineWDSchedule for weight decay scheduling.
    """
    return CosineWDSchedule(
        optimizer=optimizer,
        ref_wd=ref_wd,
        T_max=T_max,
        final_wd=final_wd,
        fix_wd_thres=fix_wd_thres,
        fix_strategy=fix_strategy
    )

class LinearEMASchedule:
    """
    Linear EMA (Exponential Moving Average) momentum scheduler.
    linearly increases momentum from start_m to final_m over the given training steps.
    """
    def __init__(self, 
        ema_start=0.996, 
        ema_end=1.0, 
        num_epochs=100, 
        iters_per_epoch=100, 
        ipe_scale=1.0
    ):
        """EMA momentum scheduler supporting linear increase.
        :param ema_start: initial momentum value
        :param ema_end: final momentum value
        :param num_epochs: total number of epochs for training
        :param iters_per_epoch: number of iterations per epoch
        :param ipe_scale: scale factor for iterations per epoch (for gradient accumulation)
        """
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.ipe_scale = ipe_scale

        self.total_steps = int(iters_per_epoch * num_epochs * ipe_scale)
        self.increment = (ema_end - ema_start) / self.total_steps
        self._step = 0

    def step(self):
        momentum = self.ema_start + self.increment * min(self._step, self.total_steps)
        self._step += 1
        return momentum

    def get_momentum(self, step):
        return self.ema_start + self.increment * min(step, self.total_steps)
    
    def current_ema(self):
        return self.get_momentum(self._step)

    def state_dict(self):
        return {
            "_step": self._step,
            "ema_start": self.ema_start,
            "ema_end": self.ema_end,
            "num_epochs": self.num_epochs,
            "iters_per_epoch": self.iters_per_epoch,
            "ipe_scale": self.ipe_scale,
            "version": 1,
        }

    def load_state_dict(self, state_dict):
        """Restore the scheduler's state from a state dict."""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
                
class LinearLambdaSchedule:
    """
    Linear Lambda scheduler for combining I-JEPA and DSeq-JEPA.
    linearly increases lambda from start_lambda to final_lambda over the given training steps.
    """
    def __init__(self, 
        lambda_start=0.0, 
        lambda_end=1.0, 
        num_epochs=100, 
        iters_per_epoch=100, 
        ipe_scale=1.0
    ):
        """EMA momentum scheduler supporting linear increase.
        :param ema_start: initial momentum value
        :param ema_end: final momentum value
        :param num_epochs: total number of epochs for training
        :param iters_per_epoch: number of iterations per epoch
        :param ipe_scale: scale factor for iterations per epoch (for gradient accumulation)
        """
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.ipe_scale = ipe_scale

        self.total_steps = int(iters_per_epoch * num_epochs * ipe_scale)
        self.increment = (lambda_end - lambda_start) / self.total_steps
        self._step = 0

    def step(self):
        lambda_ = self.lambda_start + self.increment * min(self._step, self.total_steps)
        self._step += 1
        return lambda_

    def get_lambda(self, step):
        return self.lambda_start + self.increment * min(step, self.total_steps)
    
    def current_lambda(self):
        return self.get_lambda(self._step)

    def state_dict(self):
        return {
            "_step": self._step,
            "lambda_start": self.lambda_start,
            "lambda_end": self.lambda_end,
            "num_epochs": self.num_epochs,
            "iters_per_epoch": self.iters_per_epoch,
            "ipe_scale": self.ipe_scale,
            "version": 1,
        }

    def load_state_dict(self, state_dict):
        """Restore the scheduler's state from a state dict."""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

class WarmupCosineLRSchedule(object):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        final_lr=0.,
        fix_lr_thres=-1,
        fix_strategy='const'
    ):
        """
        Warmup and cosine learning rate schedule.
        :param optimizer: optimizer to adjust the learning rate for each step
        :param warmup_steps: number of warmup steps
        :param start_lr: initial learning rate at step 0
        :param ref_lr: reference learning rate after warmup
        :param T_max: total number of steps for cosine schedule
        :param final_lr: final learning rate at the end of cosine schedule
        :param fix_lr_thres: step threshold to fix learning rate
        :param fix_strategy: strategy to fix learning rate ('const' or 'linear')
        Note: this scheduler must be called at each training **step** to update the learning rate.
        """
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.
        self.fix_lr_thres = fix_lr_thres
        self.fix_strategy = fix_strategy

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        
        elif self._step < self.fix_lr_thres or self.fix_lr_thres < 0:                    
        
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        else:
            progress = float(self.fix_lr_thres - self.warmup_steps) / float(max(1, self.T_max))
            last_cosine_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
            min_cosine_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi)))
            
            if self.fix_strategy == 'const':
                new_lr = last_cosine_lr
            elif self.fix_strategy == 'linear':
                progress = float(self._step - self.fix_lr_thres) / float(max(1, self.T_max - (self.fix_lr_thres - self.warmup_steps)))
                new_lr = last_cosine_lr - progress*(last_cosine_lr - min_cosine_lr)            
        
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        return new_lr
    
    def get_current_lr(self):
        for group in self.optimizer.param_groups:
            return group['lr']
        
    def state_dict(self):
        """Return a state dict for saving the scheduler's state."""
        return {
            'start_lr': self.start_lr,
            'ref_lr': self.ref_lr,
            'final_lr': self.final_lr,
            'warmup_steps': self.warmup_steps,
            'T_max': self.T_max,
            '_step': self._step,
            'fix_lr_thres': self.fix_lr_thres,
            'fix_strategy': self.fix_strategy,
        }

    def load_state_dict(self, state_dict):
        """Restore the scheduler's state from a state dict."""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k == 'current_step':
                self._step = v
                
class MultiStepValuesLRSchedule(object):
    """
    Multi-step / piecewise LR scheduler (step-based) compatible with the logic in setup_optim.

    This scheduler:
      - Takes milestones in units of epochs.
      - Takes lr_values defining the LR scale for each phase.
      - Is called **every training step**.
      - Converts step index -> "virtual epoch" using iterations_per_epoch (ipe).
      - Reproduces:
          * MultiStepLR(optimizer, milestones, gamma) when lr_values form a geometric sequence.
          * LambdaLR(optimizer, lr_lambda=factor) with factor(epoch) = values[k] / values[0]
            otherwise (same behavior as the original setup_optim code).
    """

    def __init__(
        self,
        optimizer,
        milestones,
        values,
        iterations_per_epoch,
        start_step=0,
    ):
        assert len(milestones) + 1 == len(values), \
            "Number of milestones must be one less than number of values"
        assert iterations_per_epoch > 0, \
            "iterations_per_epoch (ipe) must be positive"

        self.optimizer = optimizer
        self.milestones = list(milestones)
        self.values = list(values)
        self.ipe = int(iterations_per_epoch)
        self._step = int(start_step)

        # Store base learning rates from optimizer param groups
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # Check if lr_values form a geometric sequence
        ratios = [
            self.values[i + 1] / self.values[i]
            for i in range(len(self.values) - 1)
        ]
        self.is_constant_ratio = all(abs(r - ratios[0]) < 1e-12 for r in ratios)
        self.gamma = ratios[0] if self.is_constant_ratio else None
        self.v0 = self.values[0]

        # Apply LR corresponding to the initial step
        factor = self._compute_factor(self._step)
        self._apply_lr(factor)

    def _step_to_epoch(self, step):
        """
        Convert a global step index to a (possibly fractional) epoch index.
        """
        return float(step) / float(self.ipe)

    def _compute_factor(self, step):
        """
        Compute LR scaling factor for a given step.
        """
        epoch = self._step_to_epoch(step)

        if self.is_constant_ratio:
            # MultiStep-style: count how many milestones have been reached.
            # Equivalent to: gamma ** (# of m where epoch >= m)
            count = 0
            for m in self.milestones:
                if epoch >= m:
                    count += 1
            return self.gamma ** count
        else:
            # Lambda-style: choose segment by epoch, scale using values[k] / v0
            # k in [0, len(values)-1]
            k = bisect_right(self.milestones, epoch)
            return self.values[k] / self.v0

    def _apply_lr(self, factor):
        """
        Apply scaled LR to all param groups.
        """
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def step(self):
        """
        Advance one training step and update learning rate.

        Call this once per optimizer update.
        """
        self._step += 1
        factor = self._compute_factor(self._step)
        self._apply_lr(factor)
        return self.get_current_lr()

    def get_current_lr(self):
        """
        Return the current learning rate of the first param group.
        (Assumes all groups share the same schedule shape.)
        """
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        """
        Return a state dict for saving the scheduler's state.
        """
        return {
            "milestones": self.milestones,
            "values": self.values,
            "iterations_per_epoch": self.ipe,
            "_step": self._step,
            "base_lrs": self.base_lrs,
            "is_constant_ratio": self.is_constant_ratio,
            "gamma": self.gamma,
            "v0": self.v0,
        }

    def load_state_dict(self, state_dict):
        """
        Restore the scheduler's state from a state dict.

        Assumes the optimizer's param_groups have already been restored.
        """
        self.milestones = list(state_dict["milestones"])
        self.values = list(state_dict["values"])
        self.ipe = int(state_dict["iterations_per_epoch"])
        self._step = int(state_dict["_step"])
        self.base_lrs = list(state_dict["base_lrs"])
        self.is_constant_ratio = state_dict["is_constant_ratio"]
        self.gamma = state_dict["gamma"]
        self.v0 = state_dict["v0"]

        # Re-apply LR for the restored step
        factor = self._compute_factor(self._step)
        self._apply_lr(factor)
    
class CosineWDSchedule(object):
    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.,
        fix_wd_thres=-1,
        fix_strategy='const'
    ):
        """
        Cosine weight decay schedule.
        :param optimizer: optimizer to adjust the weight decay for each step
        :param ref_wd: reference weight decay at the beginning
        :param T_max: total number of steps for cosine schedule
        :param final_wd: final weight decay at the end of cosine schedule
        :param fix_wd_thres: step threshold to fix weight decay
        :param fix_strategy: strategy to fix weight decay ('const' or 'linear')
        Note: this scheduler must be called at each training **step** to update the weight decay
        """
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.
        self.fix_wd_thres = fix_wd_thres
        self.fix_strategy = fix_strategy

    def step(self):
        self._step += 1
        
        if self._step < self.fix_wd_thres or self.fix_wd_thres < 0:                    
            progress = self._step / self.T_max
            new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))
        else:
            progress = float(self.fix_wd_thres) / self.T_max
            last_cosine_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))
            if self.fix_strategy == 'const':
                new_wd = last_cosine_wd
            elif self.fix_strategy == 'linear':
                progress = float(self._step - self.fix_wd_thres) / (self.T_max - self.fix_wd_thres)
                max_cosine_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * 1))
                new_wd = last_cosine_wd + progress*(max_cosine_wd - last_cosine_wd)            
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
    
    def get_current_wd(self):
        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                return group['weight_decay']
        return None
    
    def state_dict(self):
        """Return a state dict for saving the scheduler's state."""
        return {
            'ref_wd': self.ref_wd,
            'final_wd': self.final_wd,
            'T_max': self.T_max,
            '_step': self._step,
            'fix_wd_thres': self.fix_wd_thres,
            'fix_strategy': self.fix_strategy,
        }
    def load_state_dict(self, state_dict):
        """Restore the scheduler's state from a state dict."""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
