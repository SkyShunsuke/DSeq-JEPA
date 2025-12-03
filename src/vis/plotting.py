from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import math
from PIL import Image
from tqdm import tqdm

from src.utils.log import get_logger

logger = get_logger(__name__)

def plot_attention_maps(
    images: torch.Tensor,
    attn_maps: torch.Tensor,
    inv_normalize: torch.nn.Module,
    save_path: str,
    verbose: bool,
    cmap: str = 'viridis',
    inverse_score: bool = False,
):
    """
    Plots attention maps over images and saves the visualization.

    Args:
        images (torch.Tensor): Batch of images, shape (N, C, H, W).
        attn_maps (dict): Dictionary of attention maps, each of shape (N, h, w).
        inv_normalize (torch.nn.Module): Inverse normalization transform.
        save_path (str): Path to save the visualization.
        verbose (bool): Whether to log verbose messages.
        cmap (str): Colormap to use for attention maps.
        inverse_score (bool): Whether to invert attention values for visualization.
    """
    N = images.size(0)
    K = len(attn_maps.keys())
    fig, axes = plt.subplots(N, K + 1, figsize=((K + 1) * 4, N * 4))
    if N == 1:
        axes = axes.unsqueeze(0)  # Ensure axes is 2D array even for single image
    
    for i in tqdm(range(N), desc="Plotting attention maps", disable=not verbose):
        # Inverse normalize the image
        img = inv_normalize(images[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Image')
        
        for j, (layer_name, attn_map) in enumerate(attn_maps.items()):
            attn = attn_map[i].cpu().numpy()  # (h, w)
            if inverse_score:
                attn = 1 - attn  # Invert attention for better visualization
            attn_resized = np.array(Image.fromarray(attn).resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR))
            axes[i, j + 1].imshow(img)
            axes[i, j + 1].imshow(attn_resized, cmap=cmap, alpha=0.6)
            axes[i, j + 1].axis('off')
            # axes[i, j + 1].set_title(f'Attention: {layer_name}')
        
    plt.tight_layout()
    plt.savefig(save_path)
    if verbose:
        logger = get_logger()
        logger.info(f"Saved attention visualization to {save_path}")
        
def plot_similarity_maps(
    images: torch.Tensor,
    sim_maps: torch.Tensor,
    inv_normalize: torch.nn.Module,
    save_path: str,
    verbose: bool,
    cmap: str = 'viridis',
    inverse_score: bool = False,
):
    """
    Plots similarity maps over images and saves the visualization.

    Args:
        images (torch.Tensor): Batch of images, shape (N, C, H, W).
        sim_maps (dict): Dictionary of similarity maps, each of shape (N, h, w).
        inv_normalize (torch.nn.Module): Inverse normalization transform.
        save_path (str): Path to save the visualization.
        verbose (bool): Whether to log verbose messages.
        cmap (str): Colormap to use for similarity maps.
        inverse_score (bool): Whether to invert similarity values for visualization.
    """
    N = images.size(0)
    K = len(sim_maps.keys())
    fig, axes = plt.subplots(N, K + 1, figsize=((K + 1) * 4, N * 4))
    if N == 1:
        axes = axes.unsqueeze(0)  # Ensure axes is 2D array even for single image
    
    for i in tqdm(range(N), desc="Plotting similarity maps", disable=not verbose):
        # Inverse normalize the image
        img = inv_normalize(images[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Image')
        
        for j, (layer_name, sim_map) in enumerate(sim_maps.items()):
            sim = sim_map[i].cpu().numpy()  # (h, w)
            if inverse_score:
                sim = 1 - sim  # Invert similarity for better visualization
            sim_resized = np.array(Image.fromarray(sim).resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR))
            axes[i, j + 1].imshow(img)
            axes[i, j + 1].imshow(sim_resized, cmap=cmap, alpha=0.6)
            axes[i, j + 1].axis('off')
            # axes[i, j + 1].set_title(f'Similarity: {layer_name}')
        
    plt.tight_layout()
    plt.savefig(save_path)
    if verbose:
        logger = get_logger()
        logger.info(f"Saved similarity visualization to {save_path}")
        
def plot_clustering_maps(
    images: torch.Tensor,
    cluster_maps: dict,              # layer_name -> (N, h, w) int tensor in {0..C-1}
    inv_normalize: torch.nn.Module,
    save_path: str,
    num_clusters: int,
    verbose: bool,
    cmap_name: str = 'tab20',
    alpha: float = 0.7,
    draw_boundaries: bool = True,
    *,
    # New: for consistent coloring by affinity rank
    affinity_order: dict = None,   # layer_name -> list[int] (old_label sorted by affinity desc)
    affinity_scores: dict = None,  # (optional) not used if affinity_order provided
    # New: to visualize affinity as heatmaps next to clustering overlays
    affinity_maps: dict = None,    # layer_name -> (N, h, w) float tensor; higher = more affine
    affinity_cmap_name: str = 'inferno',
    show_affinity_colorbar: bool = False,
    show_affinity_maps: bool = False,
):
    """
    Extended: If affinity_maps is provided, we render both 'cluster overlay' and
    'affinity heatmap' per layer, side-by-side. Coloring of clusters remains
    consistent by re-labeling to affinity-rank labels (0..C-1).
    """
    assert isinstance(cluster_maps, dict) and len(cluster_maps) > 0, "cluster_maps must be non-empty dict"
    N = images.size(0)
    layers = list(cluster_maps.keys())
    # reverse order
    layers = layers[::-1]
    K = len(layers)

    # --- build remap (old_label -> rank_label) per layer ---
    remap_dict: dict[str, np.ndarray] = {}
    for layer_name in layers:
        if affinity_order is not None and layer_name in affinity_order:
            order = list(affinity_order[layer_name])
            assert len(order) == num_clusters, f"affinity_order[{layer_name}] must have length {num_clusters}"
            remap = np.zeros(num_clusters, dtype=np.int32)
            for new_label, old_label in enumerate(order):
                remap[old_label] = new_label
        elif affinity_scores is not None and layer_name in affinity_scores:
            scores = np.asarray(affinity_scores[layer_name]).reshape(-1)
            assert scores.shape[0] == num_clusters, f"affinity_scores[{layer_name}] must have length {num_clusters}"
            order = np.argsort(-scores)
            remap = np.zeros(num_clusters, dtype=np.int32)
            for new_label, old_label in enumerate(order):
                remap[old_label] = new_label
        else:
            remap = np.arange(num_clusters, dtype=np.int32)
        remap_dict[layer_name] = remap

    # --- LUT for cluster overlays (by rank labels 0..C-1) ---
    cmap = cm.get_cmap(cmap_name, num_clusters)
    lut = (np.array([cmap(i) for i in range(num_clusters)])[:, :3] * 255).astype(np.uint8)  # (C,3)

    # --- layout: image + (cluster, affinity) * K ---
    has_aff = (affinity_maps is not None) and show_affinity_maps
    cols_per_layer = 2 if has_aff else 1
    total_cols = 1 + K * cols_per_layer

    fig, axes = plt.subplots(N, total_cols, figsize=(total_cols * 4, N * 4))
    if N == 1:
        axes = np.expand_dims(axes, axis=0)
    if total_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in tqdm(range(N), desc="Plotting clustering+affinity maps", disable=not verbose):
        img = inv_normalize(images[i]).permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        H, W = img.shape[:2]

        # Col 0: original image
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Image')

        # Following cols: per-layer (cluster overlay [+ affinity heatmap])
        for li, layer_name in enumerate(layers):
            # --- clustering overlay ---
            j_cluster = 1 + li * cols_per_layer
            lab_hw = cluster_maps[layer_name][i].detach().cpu().numpy().astype(np.int32)
            # re-label by affinity rank
            remap = remap_dict[layer_name]
            lab_rank = remap[np.clip(lab_hw, 0, num_clusters - 1)]

            lab_resized = np.array(
                Image.fromarray(lab_rank.astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
            ).astype(np.int32)
            lab_rgb = lut[np.clip(lab_resized, 0, num_clusters - 1)]  # (H, W, 3)

            axes[i, j_cluster].imshow(img)
            axes[i, j_cluster].imshow(lab_rgb, alpha=alpha)
            axes[i, j_cluster].axis('off')
            # axes[i, j_cluster].set_title(f'{layer_name} (clusters)')

            if draw_boundaries:
                up    = np.zeros_like(lab_resized); up[1:, :]   = lab_resized[:-1, :]
                down  = np.zeros_like(lab_resized); down[:-1, :] = lab_resized[1:,  :]
                left  = np.zeros_like(lab_resized); left[:, 1:]  = lab_resized[:, :-1]
                right = np.zeros_like(lab_resized); right[:, :-1]= lab_resized[:, 1:]
                edges = (lab_resized != up) | (lab_resized != down) | (lab_resized != left) | (lab_resized != right)
                axes[i, j_cluster].imshow(edges, cmap='gray', alpha=0.35, interpolation='nearest')

            # --- affinity heatmap (optional) ---
            if has_aff:
                assert layer_name in affinity_maps, f"affinity_maps missing layer: {layer_name}"
                j_aff = j_cluster + 1
                aff_hw = affinity_maps[layer_name][i].detach().cpu().numpy().astype(np.float32)
                # resize to image size
                aff_resized = np.array(
                    Image.fromarray(aff_hw).resize((W, H), resample=Image.BILINEAR)
                ).astype(np.float32)

                # normalize per-image for visualization stability
                amin, amax = np.nanmin(aff_resized), np.nanmax(aff_resized)
                if np.isfinite(amin) and np.isfinite(amax) and amax > amin:
                    aff_vis = (aff_resized - amin) / (amax - amin)
                else:
                    aff_vis = np.zeros_like(aff_resized)

                im = axes[i, j_aff].imshow(aff_vis, cmap=affinity_cmap_name, interpolation='nearest')
                axes[i, j_aff].axis('off')
                axes[i, j_aff].set_title(f'{layer_name} (affinity)')

                if show_affinity_colorbar and i == 0:
                    # only draw colorbar once per column to avoid clutter
                    plt.colorbar(im, ax=axes[:, j_aff], fraction=0.02, pad=0.01)

    # Legend for cluster ranks (0 = highest affinity cluster)
    legend_patches = [Patch(facecolor=cmap(i), label=f'{i}') for i in range(num_clusters)]
    ax_for_legend = axes[-1, -1]
    ax_for_legend.legend(handles=legend_patches, title="Affinity Rank", loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        logger = get_logger()
        logger.info(f"Saved clustering visualization to {save_path}")

def plot_masking(
    imgs: torch.Tensor,
    masks_enc: list,
    masks_pred: list,
    inv_normalize: torch.nn.Module,
    save_path: str,
    verbose: bool,
    cmap_name: str = 'tab20',
    feature_res: tuple = (14, 14),
    # --- fill/outline params ---
    fill_alpha: float = 0.28,         # semi-transparent fills so base image remains visible
    enc_outline_idx: int = 0,         # tab20 index for encoder outline
    pred_outline_idx: int = 2,        # tab20 index for predictor outline
    enc_linewidth: float = 5.0,
    pred_linewidth: float = 1.0,
    enc_linestyle: str = '-',         # '-' or '--' etc.
    pred_linestyle: str = '-',
    brighten: float = 1.5,            # simple gain for base image if you still find it dark
):
    """
    Draw per-mask fills (different colors within category) and category-colored outlines.
    Encoder and predictor masks are overlaid simultaneously.
    - Fills: enc masks use distinct colors; pred masks use distinct colors (separate palette offsets).
    - Outlines: enc uses a single outline color; pred uses a single (different) outline color.
    """

    # ---- basic checks ----
    assert isinstance(masks_enc, (list, tuple)) and isinstance(masks_pred, (list, tuple)), \
        "masks_enc/masks_pred must be lists (per-layer)."
    L = len(masks_enc); assert len(masks_pred) == L, "masks_enc and masks_pred must have same length (L)."
    assert imgs.dim() == 4, "imgs must be (L*B, C, H, W)."

    h_feat, w_feat = int(feature_res[0]), int(feature_res[1])
    num_patches = h_feat * w_feat

    # infer batch size B
    B = None
    for l in range(L):
        if isinstance(masks_enc[l], torch.Tensor):
            B = masks_enc[l].shape[0]; break
    if B is None:
        for l in range(L):
            if isinstance(masks_pred[l], torch.Tensor):
                B = masks_pred[l].shape[0]; break
    assert B is not None, "Cannot infer batch size B from masks."
    assert imgs.size(0) == L * B, "imgs must be concatenated as (L*B, ...), matching masks' L and B."

    # ---- color palettes ----
    # We build two independent fill palettes from tab20:
    #   - enc fills: indices [0,1,2,3,4,5,6,7,8,9] (skipping enc outline index to avoid identical color)
    #   - pred fills: indices [10,11,12,13,14,15,16,17,18,19] (skipping pred outline index if it lands here)
    # This keeps fill colors distinct across masks and consistent across samples.
    cmap = cm.get_cmap(cmap_name, 20)  # tab20 has 20 discrete colors
    enc_outline_rgba = cmap(enc_outline_idx)
    pred_outline_rgba = cmap(pred_outline_idx)

    enc_fill_idxs = [i for i in range(0, 10) if i != enc_outline_idx] or [0]
    pred_fill_idxs = [i for i in range(10, 20) if i != pred_outline_idx] or [12]

    enc_fill_rgbs = (np.array([cmap(i)[:3] for i in enc_fill_idxs]) * 255).astype(np.uint8)  # (Epal, 3)
    pred_fill_rgbs = (np.array([cmap(i)[:3] for i in pred_fill_idxs]) * 255).astype(np.uint8)  # (Ppal, 3)

    # ---- helpers ----
    def vec_to_indices(v: torch.Tensor) -> torch.Tensor:
        """Accepts binary vector (N) or index list (-1 padded). Returns LongTensor of indices in [0, N)."""
        v = v.detach().cpu().to(torch.int64).view(-1)
        if v.numel() == num_patches and ((v == 0) | (v == 1)).all():
            idx = torch.nonzero(v, as_tuple=False).view(-1)
        else:
            idx = v[v >= 0].view(-1)
            idx = idx[(idx >= 0) & (idx < num_patches)]
        return idx

    def draw_mask(ax, mask_hw: np.ndarray, fill_rgb: np.ndarray, outline_rgba, lw: float, ls: str, contour_alpha: float = 0.8):
        """Draw one mask: first semi-transparent fill, then category-colored outline."""
        if not mask_hw.any():
            return
        H_img, W_img = ax.images[0].get_array().shape[:2]  # base image size from first imshow
        mask_im = np.array(Image.fromarray(mask_hw).resize((W_img, H_img), resample=Image.NEAREST), dtype=np.uint8)

        # fill (alpha blending, minimal darkening due to small fill_alpha)
        if fill_alpha > 0:
            canvas = np.zeros((H_img, W_img, 3), dtype=np.uint8)
            canvas[mask_im.astype(bool)] = fill_rgb
            ax.imshow(canvas, alpha=fill_alpha)

        # outline using contour
        ax.contour(
            mask_im,
            levels=[0.5],
            colors=[outline_rgba],
            linewidths=lw,
            linestyles=ls,
            antialiased=True,
            alpha=contour_alpha
        )

    # ---- plotting ----
    N = imgs.size(0)
    fig, axes = plt.subplots(N, 1, figsize=(5, 4 * N))
    if N == 1:
        axes = [axes]

    for i in tqdm(range(N), desc="Plotting masking (fills + category-colored outlines)", disable=not verbose):
        l = i // B; b = i % B

        # base image (optional brighten)
        img = inv_normalize(imgs[i]).permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img * 255 * float(brighten), 0, 255).astype(np.uint8)

        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')

        # --- Encoder masks: fill colors per mask, same outline color for all enc masks ---
        if isinstance(masks_enc[l], torch.Tensor):
            E = masks_enc[l].shape[1]
            for e in range(E):
                idx = vec_to_indices(masks_enc[l][b, e])
                if idx.numel():
                    y = (idx // w_feat).numpy()
                    x = (idx %  w_feat).numpy()
                    enc_hw = np.zeros((h_feat, w_feat), dtype=np.uint8)
                    enc_hw[y, x] = 1
                    # choose fill color cycling within enc palette
                    fill_rgb = enc_fill_rgbs[e % len(enc_fill_rgbs)]
                    draw_mask(
                        ax, enc_hw, fill_rgb=fill_rgb,
                        outline_rgba=enc_outline_rgba,
                        lw=enc_linewidth, ls=enc_linestyle
                    )

        # --- Predictor masks: fill colors per mask, same outline color for all pred masks ---
        if isinstance(masks_pred[l], torch.Tensor):
            P = masks_pred[l].shape[1]
            for p in range(P):
                idx = vec_to_indices(masks_pred[l][b, p])
                if idx.numel():
                    y = (idx // w_feat).numpy()
                    x = (idx %  w_feat).numpy()
                    pred_hw = np.zeros((h_feat, w_feat), dtype=np.uint8)
                    pred_hw[y, x] = 1
                    # choose fill color cycling within pred palette
                    fill_rgb = pred_fill_rgbs[p % len(pred_fill_rgbs)]
                    draw_mask(
                        ax, pred_hw, fill_rgb=fill_rgb,
                        outline_rgba=pred_outline_rgba,
                        lw=pred_linewidth, ls=pred_linestyle
                    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        logger = get_logger()
        logger.info(f"Saved masking visualization to {save_path}")
