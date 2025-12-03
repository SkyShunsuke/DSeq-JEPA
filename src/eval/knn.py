import os
import json
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from src.models import init_target_encoder
from src.utils.dist import init_distributed_mode, get_rank, get_world_size
from src.utils.dist import is_main_process as is_main
from src.utils.log import setup_logging, get_logger
from src.dataset import make_probing_transforms, make_dataset
from src.utils.opt.optimzer import load_jepa_target_encoder_weights
from src.eval.feature_extractor import KNNEvaluatorDistributed, build_feature_bank_ddp

def main(params, args):
    """KNN function for JEPA target encoder.
    We use following words in the code:
    - target_encoder: the momentum encoder that encodes the target patch
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
        # this experiment
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = params['logging']['log_dir']
    
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
    logger.info(f"Model Info: {model}")
    
    train_transform, test_transform = make_probing_transforms(
        crop_size,
        img_size,
        interpolation=params['data']['augmentation'].get('interpolation', 3),
    )

    _, bank_loader, _ = make_dataset(
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
    _, query_loader, _ = make_dataset(
        dataset_name=params['data']['dataset_name'],
        transform=test_transform,
        batch_size=params['data']['batch_size_per_replica'],
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
    logger.info(f"Data Info: Dataset: {params['data']['dataset_name']}, #Images: {len(bank_loader.dataset)} train, {len(query_loader.dataset)} val.")
    
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    model.eval()
    
    # -- run knn evaluation
    logger.info("Starting KNN evaluation...")
    logger.info("Building feature bank...")
    verbose = params['logging'].get('verbose', True)
    knn = KNNEvaluatorDistributed(
        k=params['knn']['knn_k'],
        temperature=params['knn']['knn_t'],
        num_classes=params['knn']['num_classes'],
        feature_dim=model.module.embed_dim,
        feature_strategies=params['knn']['feature_strategy'],
    )
    logger.info(f"KNN Info: K={params['knn']['knn_k']}, T={params['knn']['knn_t']}, feature strategies: {params['knn']['feature_strategy']}")
    bank_feats, bank_labels = build_feature_bank_ddp(
        model.module,
        loader=bank_loader,
        normalize=True,
        use_amp=params['knn']['use_amp'],
        feature_strategies=knn.feature_strategies,
        verbose=verbose,
    )
    for strategy in bank_feats.keys():
        knn.set_bank(bank_feats[strategy], bank_labels[strategy], strategy)
    
    logger.info("Evaluating on query set...")
    results_dict = knn.evaluate_ddp(
        model=model.module,
        loader=query_loader,
        normalize=True,
        sim_chunk_size=params['knn'].get('sim_chunk_size', 4096),
        use_amp=params['knn']['use_amp'],
        verbose=False,
    )
    logger.info("KNN evaluation results:")
    for strategy in bank_feats.keys():
        logger.info(f"KNN Eval Strategy: {strategy} Top-1 Acc: {results_dict[strategy]['top1']:.2f}%, Top-5 Acc: {results_dict[strategy]['top5']:.2f}%")
    
    # -- log results to json file
    if is_main():
        results_file = os.path.join(log_dir, f"{model_id}_knn_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        logger.info(f"KNN evaluation results are saved to {results_file}")

    # -- close distributed process
    dist.barrier()
    dist.destroy_process_group()

    # -- end of main
    logger.info("Evaluation completed. Evaluation results are saved in %s", log_dir)