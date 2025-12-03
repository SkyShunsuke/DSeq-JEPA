import math
import torch
import torch.distributed as dist
from torch.nn.functional import normalize as F_normalize

from src.utils.dist import init_distributed_mode, get_rank, get_world_size, \
    concat_all_gather, sum_all_reduce
from src.utils.dist import is_main_process as is_main
from src.models.vision_transformer import VisionTransformer
from src.utils.log import get_logger

from tqdm import tqdm

FEATURE_STRATEGIES = [
    "lastPOOL",
    "concatPOOL4",
]

logger = get_logger()

@torch.no_grad()
def extract_features_ddp(
    model: VisionTransformer,
    loader: torch.utils.data.DataLoader,
    normalize=False,
    use_amp=False,
    feature_strategies=["lastPOOL"],
    device=None,
    verbose=True,
) -> tuple:
    """Extract features using a model in DDP mode.
    param: model: The model to use for feature extraction.
    param: loader: DataLoader providing the data to extract features from.
    param: normalize: Whether to normalize the features., NOTE: by default, features are already normalized in the call of get_intermediate_features().
    param: use_amp: Whether to use automatic mixed precision.
    param: device: The device to use for computation.

    return: A tuple of (features, labels) tensors.
    
    Example:
        feats, labels = extract_features_ddp(
            model, loader, normalize=True, use_amp=True,
            feature_strategies=["lastPOOL", "concatPOOL4"]
        )
        feats -> {
            "lastPOOL": Tensor of shape (N, D),
            "concatPOOL4": Tensor of shape (N, D'),
        }
        labels -> {
            "lastPOOL": Tensor of shape (N,),
            "concatPOOL4": Tensor of shape (N,),
        }
    """
    for strategy in feature_strategies:
        assert strategy in FEATURE_STRATEGIES, f"Unknown feature strategy: {strategy}"
    
    model_was_training = model.training
    model.eval()
    
    feats_local = {strategy: [] for strategy in feature_strategies}
    labels_local = {strategy: [] for strategy in feature_strategies}
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    device = f"cuda:{get_rank()}" if device is None else device
    
    logger.info(f"Extracting features for {len(loader)} batches on device {device}...")
    i = 0
    for batch in tqdm(loader, disable=not verbose):
        i += 1
        if len(batch) == 2:
            images, labels = batch[0], batch[1]
        elif len(batch) == 3:
            images, labels = batch[0][0], batch[0][1]
        else:
            raise ValueError("Batch must be a tuple of (images, labels) or (images, labels, _)")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            features = model.get_intermediate_features(images, names=feature_strategies, masks=None)
            
        if normalize:
            for strategy in feature_strategies:
                feats_local[strategy].append(F_normalize(features[strategy], dim=-1).float().detach())
                labels_local[strategy].append(labels.detach())
        else:
            for strategy in feature_strategies:
                feats_local[strategy].append(features[strategy].float().detach())
                labels_local[strategy].append(labels.detach())    
    
    feats_all = {
        strategy: [] for strategy in feature_strategies
    }
    for strategy in feature_strategies:
        if len(feats_local[strategy]) == 0:
            feats_local[strategy] =  torch.empty(0, getattr(model.module if hasattr(model, "module") else model, "out_dim", 0), device=device)
            labels_local[strategy] = torch.empty(0, dtype=torch.long, device=device)
        else:
            feats_local[strategy] = torch.cat(feats_local[strategy], dim=0)
            labels_local[strategy] = torch.cat(labels_local[strategy], dim=0)

        # -- collate features from all gpus
        feats_all[strategy] = concat_all_gather(feats_local[strategy])
    # -- collate labels from all gpus
    labels_all = {strategy: concat_all_gather(labels_local[strategy]) for strategy in feature_strategies}
    if model_was_training:
        model.train()
    return feats_all, labels_all

@torch.no_grad()
def build_feature_bank_ddp(
    model, loader, normalize=True, use_amp=False, device=None, feature_strategies=["lastPOOL"], verbose=True
):
    feats, labels = extract_features_ddp(
        model, loader, normalize=normalize, use_amp=use_amp,
        feature_strategies=feature_strategies, device=device, verbose=verbose
    )
    return feats, labels

@torch.no_grad()
def knn_predict(
    query_feats,
    bank_feats, 
    bank_labels,
    k: int = 200,
    temperature: float = 0.07,
    num_classes: int = 1000,
    sim_chunk_size: int = 4096,
    verbose: bool = True,
):
    """
    Perform k-NN prediction.
    Args:
        query_feats (torch.Tensor): Query features of shape (N, D).
        bank_feats (torch.Tensor): Feature bank of shape (M, D).
        bank_labels (torch.Tensor): Labels corresponding to the feature bank of shape (M,).
        k (int): Number of nearest neighbors to consider.
        temperature (float): Temperature parameter for scaling similarities.
        num_classes (int): Number of classes.
        sim_chunk_size (int): Chunk size for similarity computation to avoid OOM.
        verbose: bool = True,
    Returns:
        torch.Tensor: Predicted labels for the query features of shape (N,).
    """
    device = query_feats.device
    Nb = bank_feats.size(0)
    assert bank_labels.size(0) == Nb, "Bank features and labels must have the same number of samples."
    
    pred_top1 = torch.empty(query_feats.size(0), dtype=torch.long, device=device)
    pred_top5 = torch.empty(query_feats.size(0), 5, dtype=torch.long, device=device)

    # -- process in chunks to avoid OOM
    start = 0
    with tqdm(total=query_feats.size(0), disable=not verbose, desc="KNN Predict") as pbar:
        while start < query_feats.size(0):
            end = min(start + sim_chunk_size, query_feats.size(0))
            q = query_feats[start:end]  # (Bq, D)
            
            # -- compute similarity
            sim = q @ bank_feats.t()  # (Bq, Nb)
            sim_vals, sim_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True) # (Bq, k)
            neighbors = bank_labels[sim_idx]  # (Bq, k)
            
            # -- soft voting
            weights = torch.exp(sim_vals / temperature)  # (Bq, k)
            
            # -- voting for each class
            Bq = q.size(0)
            class_votes = torch.zeros(Bq, num_classes, device=device)  # (Bq, C)
            class_votes.scatter_add_(1, index=neighbors, src=weights)
            
            # -- predictions
            top1 = class_votes.argmax(dim=1)  # (Bq,)
            top5 = torch.topk(class_votes, k=min(5, num_classes), dim=1, largest=True, sorted=True).indices  # (Bq, 5)
            
            pred_top1[start:end] = top1
            pred_top5[start:end, :] = top5
            start = end
            pbar.update(end - start)
    return pred_top1, pred_top5

class KNNEvaluatorDistributed:
    def __init__(self, k=200, temperature=0.07, num_classes=1000, feature_dim=None, bank_capacity=None, feature_strategies=["lastPOOL", "concatPOOL4"]):
        """
        KNN Evaluator in Distributed setting.
        
        param k: Number of nearest neighbors.
        param temperature: Temperature for similarity scaling.
        param num_classes: Number of classes.
        param feature_dim: Dimension of the features.
        param bank_capacity: Capacity of the feature bank.
        param feature_strategies: List of feature strategies to evaluate.
        """
        self.k = k
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.bank_capacity = bank_capacity
        self.feature_strategies = feature_strategies

        self.bank_feats = {strategy: None for strategy in feature_strategies}
        self.bank_labels = {strategy: None for strategy in feature_strategies}

    def check_strategy(self, strategy: str):
        assert strategy in self.feature_strategies, f"Unknown feature strategy: {strategy}, available: {self.feature_strategies}"
    
    def check_bank(self):
        for strategy in self.feature_strategies:
            self.check_strategy(strategy)
            assert self.bank_feats[strategy] is not None and self.bank_labels[strategy] is not None, f"Call set_bank()/update_bank() first, for {strategy}."
            assert strategy in self.bank_feats and strategy in self.bank_labels, f"Feature bank for strategy {strategy} is not set."
            assert self.bank_feats[strategy].size(0) == self.bank_labels[strategy].size(0), "Bank size mismatch."
    
    def device(self):
        """Get the device of the feature bank."""
        self.check_bank()
        for strategy in self.feature_strategies:
            return self.bank_feats[strategy].device
    
    @torch.no_grad()
    def set_bank(self, feats: torch.Tensor, labels: torch.Tensor, strategy: str):
        """Replace the feature bank.
        param feats: Feature tensor of shape (N, D).
        param labels: Label tensor of shape (N,).
        param strategy: Feature strategy to set.
        """
        if self.bank_capacity is not None and feats.size(0) > self.bank_capacity:
            feats = feats[-self.bank_capacity:]
            labels = labels[-self.bank_capacity:]
        self.bank_feats[strategy] = feats.contiguous()
        self.bank_labels[strategy] = labels.contiguous()
        logger.info(f"Feature bank for strategy {strategy} set with {feats.size(0)} samples.")

    @torch.no_grad()
    def update_bank(self, feats_new: torch.Tensor, labels_new: torch.Tensor, strategy: str):
        """Update the feature bank by appending new features.
        param feats_new: New feature tensor of shape (N, D).
        param labels_new: New label tensor of shape (N,).
        param strategy: Feature strategy to update.
        """
        if self.bank_feats is None:
            return self.set_bank(feats_new, labels_new, strategy)

        feats = torch.cat([self.bank_feats[strategy], feats_new], dim=0)
        labels = torch.cat([self.bank_labels[strategy], labels_new], dim=0)

        if self.bank_capacity is not None and feats.size(0) > self.bank_capacity:
            feats = feats[-self.bank_capacity:]
            labels = labels[-self.bank_capacity:]

        self.bank_feats[strategy] = feats.contiguous()
        self.bank_labels[strategy] = labels.contiguous()

    @torch.no_grad()
    def evaluate_ddp(self, model, loader, normalize=False, use_amp=True, sim_chunk_size=4096, device=None, norm_fn=F_normalize, verbose=True):
        """Evaluate using k-NN in DDP mode.
        param model: The model to extract features.
        param loader: DataLoader providing the data to evaluate.
        param normalize: Whether to normalize the features.
        param use_amp: Whether to use automatic mixed precision.
        param sim_chunk_size: Chunk size for similarity computation to avoid OOM.
        param device: The device to use for computation.
        param norm_fn: Normalization function to apply to features.
        param verbose: Whether to display a progress bar.
        
        return: A dict mapping feature strategy to (top1, top5) accuracy tuple.
        """
        self.check_bank()

        bank_feats = self.bank_feats
        bank_labels = self.bank_labels
        num_types = len(self.feature_strategies)
        
        if device is None:
            device = self.device()
            
        model_was_training = model.training
        model.eval()
        
        correct1_local = torch.zeros((num_types, 1), dtype=torch.long, device=device)
        correct5_local = torch.zeros((num_types, 1), dtype=torch.long, device=device)
        total_local = torch.zeros((num_types, 1), dtype=torch.long, device=device)

        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        
        for batch in loader:
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, _ = batch
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                features = model.get_intermediate_features(images, names=self.feature_strategies, masks=None)
                assert features.keys() == bank_feats.keys(), "Feature strategies mismatch between query and bank."
                
            for idx, strategy in enumerate(self.feature_strategies):
                if normalize:
                    features[strategy] = norm_fn(features[strategy], dim=-1)
                
                pred_top1, pred_top5 = knn_predict(
                    features[strategy],
                    bank_feats[strategy],
                    bank_labels[strategy],
                    k=self.k,
                    temperature=self.temperature,
                    num_classes=self.num_classes,
                    sim_chunk_size=sim_chunk_size,
                    verbose=verbose,
                )

                correct1_local[idx] += (pred_top1 == labels).sum()
                correct5_local[idx] += (pred_top5 == labels.unsqueeze(dim=1)).any(dim=1).sum()
                total_local[idx] += torch.tensor(labels.size(0), dtype=torch.long, device=device)
                
        # -- aggregate results from all processes
        results = {}
        for idx, strategy in enumerate(self.feature_strategies):
            correct_top1 = sum_all_reduce(correct1_local[idx].clone())
            correct_top5 = sum_all_reduce(correct5_local[idx].clone())
            total = sum_all_reduce(total_local[idx].clone())
            top1 = (correct_top1 / total * 100.0).item() if total.item() > 0 else 0.0
            top5 = (correct_top5 / total * 100.0).item() if total.item() > 0 else 0.0
            results[strategy] = {"top1": top1, "top5": top5}
            
        if model_was_training:
            model.train()
            
        return results