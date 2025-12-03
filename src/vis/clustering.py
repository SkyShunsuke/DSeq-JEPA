import os
import math
import torch
from torch import nn, Tensor
import torchvision.transforms as transforms
import yaml
from src.utils.tensor.kernel import GaussianBlur

from src.models.vision_transformer import VisionTransformer
from src.utils.log import get_logger
from src.vis.plotting import plot_clustering_maps
from src.vis.utils import concat_results_dict
from src.vis.similarity import get_similarity_maps
from src.vis.attention import get_attention_maps

MAP_TYPE = ['attn', 'similarity']

logger = get_logger(__name__)

def check_methods(
    map_type: str,
    layers: list,
):
    assert map_type in MAP_TYPE, f"Unsupported map type: {map_type}, supported types are: {MAP_TYPE}"

@torch.no_grad()
def sparsify_affinity(M: Tensor, topk: int) -> Tensor:
    """
    Row-wise top-k sparsification + symmetrization.
    M: (B, N, N), non-negative and roughly symmetric is assumed.
    """
    if topk is None or topk <= 0 or topk >= M.size(-1):
        return M
    B, N, _ = M.shape
    # keep topk (excluding diagonal)
    M = M.clone()
    # ensure diag is zero before topk (optional)
    eye = torch.eye(N, device=M.device, dtype=M.dtype)
    M = M * (1 - eye)  # zero diag
    vals, idx = torch.topk(M, k=topk, dim=-1)  # (B, N, topk)
    keep = torch.zeros_like(M)
    ar = torch.arange(N, device=M.device).unsqueeze(-1).expand(B, N, topk)
    keep.scatter_(-1, idx, vals)  # keep topk weights
    # symmetrize
    keep = 0.5 * (keep + keep.transpose(-1, -2))
    return keep

@torch.no_grad()
def kmeans_torch(X: Tensor, k: int, n_init: int = 1, max_iter: int = 50, eps: float = 1e-8):
    """
    Batched k-means (Lloyd) in torch.
    X: (B, N, D)  -> returns labels: (B, N)
    """
    B, N, D = X.shape
    best_labels = None
    best_inertia = torch.full((B,), float("inf"), device=X.device, dtype=X.dtype)

    def kpp_init(X, k):
        B, N, D = X.shape
        centers = torch.empty((B, k, D), device=X.device, dtype=X.dtype)
        # first center uniform
        idx0 = torch.randint(0, N, (B,), device=X.device)
        centers[:, 0, :] = X[torch.arange(B, device=X.device), idx0, :]
        # next centers via D^2 sampling
        for c in range(1, k):
            d2 = torch.cdist(X, centers[:, :c, :], p=2) ** 2  # (B, N, c)
            min_d2, _ = d2.min(dim=-1)
            prob = (min_d2 + 1e-12) / (min_d2.sum(dim=1, keepdim=True) + 1e-12)
            idx = torch.multinomial(prob, 1).squeeze(-1)
            centers[:, c, :] = X[torch.arange(B, device=X.device), idx, :]
        return centers

    for _ in range(n_init):
        centers = kpp_init(X, k)
        for _ in range(max_iter):
            dists = torch.cdist(X, centers, p=2)  # (B, N, k)
            labels = dists.argmin(dim=-1)         # (B, N)
            # update
            new_centers = torch.zeros_like(centers)
            for c in range(k):
                mask = (labels == c).float().unsqueeze(-1)  # (B, N, 1)
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                new_centers[:, c, :] = (mask * X).sum(dim=1) / denom.squeeze(1)
            shift = (new_centers - centers).norm(dim=-1).mean()
            centers = new_centers
            if shift < 1e-6:
                break

        # inertia
        dists = torch.cdist(X, centers, p=2)
        min_d = dists.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        inertia = (min_d ** 2).sum(dim=-1)  # (B,)

        improved = inertia < best_inertia
        if best_labels is None:
            best_labels = labels.clone()
            best_inertia = inertia
        else:
            best_labels = torch.where(improved.unsqueeze(-1), labels, best_labels)
            best_inertia = torch.where(improved, inertia, best_inertia)

    return best_labels

@torch.no_grad()
def kway_ncut(M: Tensor, k: int, use_lobpcg: bool = False, eps: float = 1e-8):
    """
    Spectral clustering by minimizing k-way Ncut via L_sym eigenvectors + k-means.
    M: (B, N, N) non-negative, (approximately) symmetric affinity.
    returns labels: (B, N)
    """
    B, N, N2 = M.shape
    assert N == N2, "M must be square"
    # symmetrize & clamp
    M = torch.clamp(0.5 * (M + M.transpose(-1, -2)), min=0.0)
    # degree
    deg = M.sum(dim=-1)  # (B, N)
    # avoid isolated: tiny self-loop
    iso = deg <= eps
    if iso.any():
        eye = torch.eye(N, device=M.device, dtype=M.dtype)
        M = M + eps * eye.unsqueeze(0)
        deg = M.sum(dim=-1)

    d_inv_sqrt = (deg + eps).rsqrt()
    A = d_inv_sqrt.unsqueeze(-1) * M * d_inv_sqrt.unsqueeze(-2)  # (B, N, N)
    I = torch.eye(N, device=M.device, dtype=M.dtype).expand(B, -1, -1)
    Lsym = (I - A)
    Lsym = 0.5 * (Lsym + Lsym.transpose(-1, -2))  # enforce symmetry

    if not use_lobpcg:
        evals, evecs = torch.linalg.eigh(Lsym)     # (B, N), (B, N, N)
        U = evecs[:, :, :k]                        # (B, N, k)
    else:
        # lobpcg expects (N,N), so loop B (usually small). Use a small random init.
        U_list = []
        for b in range(B):
            # random init (N, k)
            X0 = torch.randn(N, k, device=M.device, dtype=M.dtype)
            # Note: torch.lobpcg works on symmetric positive definite; Lsym is PSD.
            # largest=False isn't supported in lobpcg; we approximate by providing X0 and rely on convergence to smallest.
            # If needed, shift-invert or use eigh on small N.
            evals_b, evecs_b = torch.lobpcg(Lsym[b], X0, niter=200)
            # take the k columns we solved for
            U_list.append(evecs_b)  # (N, k)
        U = torch.stack(U_list, dim=0)  # (B, N, k)

    # row-normalize & k-means
    U = U / (U.norm(dim=-1, keepdim=True) + eps)
    labels = kmeans_torch(U, k=k, n_init=1, max_iter=50).to(torch.long)
    return labels

@torch.no_grad()
def infer_hw(model, imgs: Tensor, N_tokens: int):
    """
    Try to infer (h, w) for patch tokens (excluding CLS).
    Priority:
      1) model.patch_embed.grid_size (common in timm ViT)
      2) imgs.shape & model.patch_embed.patch_size
      3) sqrt(N)
    """
    # 1) grid_size
    grid = getattr(getattr(model, 'patch_embed', None), 'grid_size', None)
    if grid is not None:
        if isinstance(grid, (tuple, list)) and len(grid) == 2:
            return int(grid[0]), int(grid[1])
        if isinstance(grid, int):
            return int(grid), int(grid)

    # 2) infer from image shape and patch size
    patch_size = getattr(getattr(model, 'patch_embed', None), 'patch_size', None)
    if patch_size is not None and isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        H, W = imgs.shape[-2], imgs.shape[-1]
        ph, pw = patch_size
        if H % ph == 0 and W % pw == 0 and (H // ph) * (W // pw) == N_tokens:
            return H // ph, W // pw

    # 3) fallback: square
    s = int(round(math.sqrt(N_tokens)))
    return s, max(1, N_tokens // s)

@torch.no_grad()
def extract_affinity_from_entry(entry) -> Tensor:
    """
    Accepts either a Tensor (B,N,N) or a dict containing 'affinity' tensor.
    """
    if isinstance(entry, dict):
        if 'affinity' in entry and isinstance(entry['affinity'], torch.Tensor):
            return entry['affinity']
        raise ValueError("Expected dict with key 'affinity' -> Tensor(B,N,N).")
    if isinstance(entry, torch.Tensor):
        if entry.dim() == 3 and entry.size(-1) == entry.size(-2):
            return entry
    raise ValueError(f"map_dict[layer] must be Tensor(B,N,N) or dict({'affinity': Tensor(B,N,N)})., got: {type(entry)}")


@torch.no_grad()
def cluster_maps_by_layer(
    model,
    imgs: Tensor,                 # (B, C, H, W) — for (h,w) inference only
    map_dict: dict,               # layer -> (B, N, N) or dict{'affinity': ...}
    num_clusters: int,
    *,
    sparsify_graph: bool = False,
    topk: int = 20,
    use_lobpcg: bool = False,
    inverse_affinity: bool = False,
) -> dict:
    """
    For each layer in map_dict, run k-way Ncut and return label maps of shape (B, h, w).
    """
    out = {}
    sample_layer = next(iter(map_dict.keys()))
    M0 = extract_affinity_from_entry(map_dict[sample_layer])
    B, N, _ = M0.shape
    h, w = infer_hw(model, imgs, N_tokens=N)

    for layer, entry in map_dict.items():
        M = extract_affinity_from_entry(entry)  # (B, N, N)

        if inverse_affinity:
            M = 1 - M

        if sparsify_graph:
            M = sparsify_affinity(M, topk=topk)

        labels_bn = kway_ncut(M, k=num_clusters, use_lobpcg=use_lobpcg)  # (B, N)
        labels_bhw = labels_bn.view(B, h, w)                             # (B, h, w)
        out[layer] = labels_bhw
    return out


@torch.no_grad()
def compute_affinity_order(
    model: "VisionTransformer",
    images: torch.Tensor,              # (N, C, H, W) used in the visualization
    cluster_maps: dict,                # layer -> (N, h, w) int labels
    *,
    map_type: str,
    vis_config: dict,
    num_clusters: int,
    sparsify_graph: bool,
    topk: int,
    inverse_affinity: bool = False,
) -> dict:
    """
    Compute per-layer affinity-based cluster ranking (desc) to ensure consistent coloring.

    Affinity definition:
      - Use the corresponding (B,N,N) affinity/similarity/attention matrix M used for clustering.
      - Node affinity = degree = sum_j M_ij
      - Cluster affinity score = average of node degrees over tokens assigned to that cluster.
      - Order clusters by descending score to produce a consistent affinity_order.

    Notes:
      - Applies the same preprocessing as clustering: inverse_affinity / sparsify_graph.
      - If a cluster has no tokens, assign -inf so it is ordered last.
    """
    # Recompute map_dict for the provided images (to align with visualization set)
    if map_type == 'similarity':
        map_dict, _ = get_similarity_maps(model, images, native_sim=True, **vis_config)  # layer -> (N,N) per sample
    elif map_type == 'attn':
        map_dict, _ = get_attention_maps(model, images, native_attn=True, **vis_config)
    else:
        raise NotImplementedError(f"Unsupported map type: {map_type}")

    # Prepare outputs
    affinity_order = {}   # layer -> list[int] (old_label sorted by affinity desc)

    # Determine token grid size (N tokens) and flatten labels to (N, N_tokens)
    sample_layer = next(iter(map_dict.keys()))
    M0 = extract_affinity_from_entry(map_dict[sample_layer])
    B, Ntok, _ = M0.shape

    # Flatten labels (N, h, w) -> (N, Ntok); this must match ViT tokenization grid
    # NOTE: cluster_maps were created with the same infer_hw, so reshape is consistent.
    for layer, labels_bhw in cluster_maps.items():
        # Retrieve affinity matrix (B, Ntok, Ntok)
        M = extract_affinity_from_entry(map_dict[layer]).clone()

        # Same preprocessing as clustering
        if inverse_affinity:
            M = 1 - M
        if sparsify_graph:
            M = sparsify_affinity(M, topk=topk)

        # Node degree: (B, Ntok)
        deg = M.sum(dim=-1)

        # Labels: (B, Ntok)
        B_, h, w = labels_bhw.shape
        assert B_ == B, "Batch size mismatch between affinity maps and cluster labels."
        labels_bn = labels_bhw.view(B, -1).to(deg.device)  # (B, Ntok)

        # Compute cluster scores: mean degree over tokens in the cluster
        scores = torch.full((num_clusters,), float('-inf'), device=deg.device)
        for c in range(num_clusters):
            mask = (labels_bn == c)
            if mask.any():
                # Average degree over all tokens assigned to c
                scores[c] = deg[mask].mean()

        # Sort by descending affinity (high -> low) to get order of original labels
        order = torch.argsort(scores, descending=True)  # (C,)
        affinity_order[layer] = order.detach().cpu().tolist()

    return affinity_order

@torch.no_grad()
def compute_affinity_order_and_maps(
    model: "VisionTransformer",
    images: torch.Tensor,              # (N, C, H, W) used in the visualization
    cluster_maps: dict,                # layer -> (N, h, w) int labels
    *,
    map_type: str,
    vis_config: dict,
    num_clusters: int,
    sparsify_graph: bool,
    topk: int,
    inverse_affinity: bool = False,
) -> tuple:
    """
    Compute:
      (1) affinity_order: per-layer list[int] of original labels sorted by descending affinity
      (2) affinity_maps:  per-layer (N, h, w) tensor of per-token 'affinity' (node degree)

    Affinity definition:
      - Start from the (B,N,N) matrix M used for clustering (similarity/attention).
      - Apply same preprocessing as clustering: inverse_affinity / sparsify_graph.
      - Node affinity (per token) = degree = sum_j M_ij.
      - For each image, reshape degree to (h,w) to form an affinity heatmap.
      - Cluster score = average affinity of tokens assigned to that cluster (across batch).

    Returns:
      affinity_order: dict[layer] -> list[int]
      affinity_maps:  dict[layer] -> Tensor (B, h, w), float
    """
    # Recompute map_dict for the same images we will visualize
    if map_type == 'similarity':
        map_dict, _ = get_similarity_maps(model, images, native_sim=True, **vis_config)
    elif map_type == 'attn':
        map_dict, _ = get_attention_maps(model, images, native_attn=True, **vis_config)
    else:
        raise NotImplementedError(f"Unsupported map type: {map_type}")

    # Infer token grid size
    sample_layer = next(iter(map_dict.keys()))
    M0 = extract_affinity_from_entry(map_dict[sample_layer])
    B, Ntok, _ = M0.shape
    # (h,w) inferred exactly as in clustering
    h, w = infer_hw(model, images, N_tokens=Ntok)

    affinity_order: dict[str, list[int]] = {}
    affinity_maps: dict[str, torch.Tensor] = {}

    for layer, labels_bhw in cluster_maps.items():
        M = extract_affinity_from_entry(map_dict[layer]).clone()

        if inverse_affinity:
            M = 1 - M
        if sparsify_graph:
            M = sparsify_affinity(M, topk=topk)

        # (B, Ntok): per-token degree
        deg = M.sum(dim=-1)

        # reshape per image to (B, h, w) for visualization
        aff_bhw = deg.view(B, h, w).contiguous()
        affinity_maps[layer] = aff_bhw

        # cluster scores by averaging over tokens assigned to each cluster
        labels_bn = labels_bhw.view(B, -1).to(deg.device)  # (B, Ntok)
        scores = torch.full((num_clusters,), float('-inf'), device=deg.device)
        for c in range(num_clusters):
            mask = (labels_bn == c)
            if mask.any():
                scores[c] = deg[mask].mean()

        order = torch.argsort(scores, descending=True)
        affinity_order[layer] = order.detach().cpu().tolist()

    return affinity_order, affinity_maps


def save_clustering_visualization(
    train_dict: dict,
    val_dict: dict,
    inv_normalize: transforms.Normalize,
    save_dir: str,
    vis_config: dict,
    verbose: bool,
    *,
    train_affinity_order: dict= None,
    val_affinity_order: dict = None,
    train_affinity_maps: dict = None,
    val_affinity_maps: dict = None,
):  
    # -- save train (clusters + affinity)
    train_save_path = os.path.join(save_dir, 'train_clustering_visualization.png')
    plot_clustering_maps(
        train_dict['imgs'],
        train_dict['clusters'],
        inv_normalize,
        train_save_path,
        vis_config.get('num_clusters', 5),
        verbose,
        cmap_name=vis_config.get('cmap', 'tab20'),
        affinity_order=train_affinity_order,
        affinity_maps=train_affinity_maps,
        affinity_cmap_name=vis_config.get('affinity_cmap', 'magma'),
        show_affinity_colorbar=vis_config.get('show_affinity_colorbar', False),
        show_affinity_maps=vis_config.get('show_maps', False),
    )
    logger.info(f"Saved train clustering visualization to {train_save_path}")
    
    # -- save val (clusters + affinity)
    val_save_path = os.path.join(save_dir, 'val_clustering_visualization.png')
    plot_clustering_maps(
        val_dict['imgs'],
        val_dict['clusters'],
        inv_normalize,
        val_save_path,
        vis_config.get('num_clusters', 5),
        verbose,
        cmap_name=vis_config.get('cmap', 'tab20'),
        affinity_order=val_affinity_order,
        affinity_maps=val_affinity_maps,
        affinity_cmap_name=vis_config.get('affinity_cmap', 'magma'),
        show_affinity_colorbar=vis_config.get('show_affinity_colorbar', False),
        show_affinity_maps=vis_config.get('show_maps', False)
    )
    
    # -- save config
    config_save_path = os.path.join(save_dir, 'clustering_visualization_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vis_config, f)
    
    logger.info(f"Saved val clustering visualization to {val_save_path}")

def visualize_clustering(
    model: "VisionTransformer",
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    inv_normalize,
    device: torch.device,
    log_dir: str,
    num_samples: int,
    vis_config: dict,
    verbose: bool,
):
    assert model.training is False, "Model should be in eval mode for visualization."
    map_type = vis_config.get('map_type', 'similarity')
    layers = vis_config.get('layers', [11])
    C = vis_config.get('num_clusters', 5)
    use_lobpcg = vis_config.get('use_lobpcg', False)
    sparsify_graph = vis_config.get('sparsify_graph', False)
    topk = vis_config.get('sparsify_topk', 20)
    inverse_affinity = vis_config.get('inverse_affinity', False)

    check_methods(map_type, layers)

    current_num = 0
    train_dict = {'clusters': {}, 'imgs': torch.tensor([])}

    # -------- Train split: cluster + collect imgs --------
    for batch in train_loader:
        images, _ = batch
        images = images.to(device, non_blocking=True)

        if map_type == 'similarity':
            map_dict, imgs = get_similarity_maps(model, images, native_sim=True, **vis_config)
        elif map_type == 'attn':
            map_dict, imgs = get_attention_maps(model, images, native_attn=True, **vis_config)
        else:
            raise NotImplementedError(f"Unsupported map type: {map_type}")

        clustered = cluster_maps_by_layer(
            model,
            imgs=imgs,
            map_dict=map_dict,
            num_clusters=C,
            sparsify_graph=sparsify_graph,
            topk=topk,
            use_lobpcg=use_lobpcg,
            inverse_affinity=inverse_affinity,
        )
        train_dict['clusters'] = concat_results_dict(train_dict['clusters'], clustered)
        train_dict['imgs'] = torch.cat([train_dict['imgs'], imgs], dim=0)

        current_num += images.size(0)
        if current_num >= num_samples:
            break

    assert current_num >= num_samples, "Not enough samples in train_loader for visualization."
    for k in train_dict['clusters']:
        train_dict['clusters'][k] = train_dict['clusters'][k][:num_samples]
    train_dict['imgs'] = train_dict['imgs'][:num_samples]

    # -------- Val split --------
    current_num = 0
    val_dict = {'clusters': {}, 'imgs': torch.tensor([])}
    for batch in val_loader:
        images, _ = batch
        images = images.to(device, non_blocking=True)

        if map_type == 'similarity':
            map_dict, imgs = get_similarity_maps(model, images, native_sim=True, **vis_config)
        elif map_type == 'attn':
            map_dict, imgs = get_attention_maps(model, images, native_attn=True, **vis_config)
        else:
            raise NotImplementedError(f"Unsupported map type: {map_type}")

        clustered = cluster_maps_by_layer(
            model,
            imgs=imgs,
            map_dict=map_dict,
            num_clusters=C,
            sparsify_graph=sparsify_graph,
            topk=topk,
            use_lobpcg=use_lobpcg,
            inverse_affinity=inverse_affinity,
        )
        val_dict['clusters'] = concat_results_dict(val_dict['clusters'], clustered)
        val_dict['imgs'] = torch.cat([val_dict['imgs'], imgs], dim=0)

        current_num += images.size(0)
        if current_num >= num_samples:
            break

    assert current_num >= num_samples, "Not enough samples in val_loader for visualization."
    for k in val_dict['clusters']:
        val_dict['clusters'][k] = val_dict['clusters'][k][:num_samples]
    val_dict['imgs'] = val_dict['imgs'][:num_samples]

    # -------- Compute affinity-based order + maps (for both splits) --------
    train_affinity_order, train_affinity_maps = compute_affinity_order_and_maps(
        model=model,
        images=train_dict['imgs'].to(device, non_blocking=True),
        cluster_maps=train_dict['clusters'],
        map_type=map_type,
        vis_config=vis_config,
        num_clusters=C,
        sparsify_graph=sparsify_graph,
        topk=topk,
        inverse_affinity=inverse_affinity,
    )
    val_affinity_order, val_affinity_maps = compute_affinity_order_and_maps(
        model=model,
        images=val_dict['imgs'].to(device, non_blocking=True),
        cluster_maps=val_dict['clusters'],
        map_type=map_type,
        vis_config=vis_config,
        num_clusters=C,
        sparsify_graph=sparsify_graph,
        topk=topk,
        inverse_affinity=inverse_affinity,
    )

    # -------- Save visualization results (clusters + affinity) --------
    save_dir = os.path.join(log_dir, 'clustering_visualization')
    os.makedirs(save_dir, exist_ok=True)
    save_clustering_visualization(
        train_dict,
        val_dict,
        inv_normalize,
        save_dir,
        vis_config,
        verbose,
        train_affinity_order=train_affinity_order,
        val_affinity_order=val_affinity_order,
        train_affinity_maps=train_affinity_maps,
        val_affinity_maps=val_affinity_maps,
    )
