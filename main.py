import argparse
import os
import yaml
import copy
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
from src.utils.opt.scheduler import get_cosine_wd_scheduler, get_ema_scheduler, get_warmup_cosine_lr_scheduler
from src.utils.opt.scaler import get_gradient_scaler
from src.utils.opt.optimzer import load_checkpoint, save_jepa_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="x-JEPA all-in-one")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/ijepa/pretrain/vitb16_in1k.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pretrain",
        required=True,
        choices=["pretrain", "classification", "detection", "segmentation", "visualization", "masking", "knn", "analysis"],
    )
    
    ### Loaded from torchrun
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    return args

def main(params, args):
    """Main function for x-JEPA.
    This function handles different tasks such as pretraining, classification, detection, segmentation, and visualization.
    It parses the given task and calls the corresponding function.
    """
    task = args.task
    if task == "pretrain":
        framework = params['framework']['name']
        if framework == "ijepa":
            from src.frameworks.ijepa.pretrain import main as ijepa_pretrain
            ijepa_pretrain(params, args)
        elif framework == "dseqjepa":
            from src.frameworks.dseqjepa.pretrain import main as dseqjepa_pretrain
            dseqjepa_pretrain(params, args)
        elif framework == "cjepa":
            from src.frameworks.cjepa.pretrain import main as cjepa_pretrain
            cjepa_pretrain(params, args)
        else:
            raise NotImplementedError(f"Pretraining for framework {framework} is not implemented.")
    elif task == "classification":
        framework = params['framework']['name']
        if framework == "probing":
            from src.eval.linear_probing import main as probing_main
            probing_main(params, args)
        elif framework == "knn":
            from src.eval.knn import main as knn_main
            knn_main(params, args)
        else:
            raise NotImplementedError(f"Classification for framework {framework} is not implemented.")
    elif task == "detection":
        raise NotImplementedError("Detection task is not implemented yet.")
    elif task == "segmentation":
        raise NotImplementedError("Segmentation task is not implemented yet.")
    elif task == "visualization":
        from src.vis.visualize import main as visualization_main
        visualization_main(params, args)
    elif task == "masking":
        from src.vis.masking import main as masking_main
        masking_main(params, args)
    elif task == "analysis":
        from src.analysis.prediction_loss import main as loss_main
        loss_main(params, args)
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    main(params, args)