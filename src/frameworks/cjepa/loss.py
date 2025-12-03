import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.dist import concat_all_gather_with_grad
    
def variance_loss(z, eps: float = 1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1.0 - std))
    return std_loss

def convariance_loss(z):
    N, D = z.size()
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)  # (D, D)
    cov_loss = off_diagonal(cov).pow(2).sum() / D
    return cov_loss
    
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Extract off-diagonal elements of a square matrix.
    params: x (torch.Tensor): Input square matrix of shape (D, D).
    returns: torch.Tensor: Flattened off-diagonal elements of shape (D*(D-1),).
    """
    assert (x.dim() == 2 and x.size(0) == x.size(1)), "Input must be a square matrix"
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def encode_projector(embeds, projector):
    """Encode embeddings through the projector MLP.
    params: embeds (torch.Tensor): Input embeddings of shape (B, N, M, D).
    params: projector (nn.Module): MLP projector network.
    returns: torch.Tensor: Projected embeddings of shape (B, N, D').
    """
    assert embeds.dim() == 4, "Expected input shape (B, N, M, D)"
    B, N, M, D = embeds.shape
    pooled_embeds = embeds.mean(dim=2)  # Global average pooling over M
    projected = projector(pooled_embeds)
    return projected

def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,  # embeds on local rank
    z1_global: torch.Tensor,
    z2_global: torch.Tensor,  # embeds on all gathered ranks
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
    compute_invariance: bool = False,
):
    """Compute VICReg loss between two sets of embeddings z1 and z2. 
    For accurate estimation of batch statistics, z1_global and z2_global are used for variance and covariance terms.
    params:
        z1 (torch.Tensor): Embeddings from view 1 on local rank of shape (B, D).
        z2 (torch.Tensor): Embeddings from view 2 on local rank of shape (B, D).
        z1_global (torch.Tensor): Embeddings from view 1 gathered from all ranks of shape (B_total, D).
        z2_global (torch.Tensor): Embeddings from view 2 gathered from all ranks of shape (B_total, D).
        sim_coeff (float): Coefficient for the similarity loss term.
        var_coeff (float): Coefficient for the variance loss term.
        cov_coeff (float): Coefficient for the covariance loss term.
        eps (float): Small constant for numerical stability.
    returns:
        tuple: (vicreg_loss, sim_loss, var_loss, cov_loss)
    """
    # --- 1. invariance (similarity) loss with local batch
    if compute_invariance:
        sim_loss = F.mse_loss(z1, z2)
    else:
        sim_loss = 0.0
    
    # --- 2. variance loss with global batch
    std_loss = variance_loss(z1_global, eps) + variance_loss(z2_global, eps)
    
    # --- 3. covariance loss with global batch
    cov_loss = convariance_loss(z1_global) + convariance_loss(z2_global)
    
    loss = sim_coeff * sim_loss + var_coeff * std_loss + cov_coeff * cov_loss
    return loss

def cjepa_loss_original(
    target_embeds: torch.Tensor,
    pred_target_embeds: torch.Tensor,
    projector: nn.Module = None,
    beta_sim: float = 25,
    beta_var: float = 25,
    beta_cov: float = 1.0,
    eps: float = 1e-4,
):
    """[Slow Original Paper Implementation] Compute CJepa loss between target embeddings and context embeddings.
    params: target_embeds (torch.Tensor): Target embeddings of shape (B, N, Mc, D) without predictor.
    params: pred_target_embeds (torch.Tensor): Predicted target embeddings of shape (B, N, Mc, D) from predictor.
    params: beta_sim (float): Coefficient for similarity loss.
    params: beta_var (float): Coefficient for variance loss.
    params: beta_cov (float): Coefficient for covariance loss.
    params: eps (float): Small constant for numerical stability.
    returns: torch.Tensor: Computed CJepa loss.
    """
    # --- Projection through MLP projector if provided
    if projector is not None:
        pred_target_embeds = encode_projector(pred_target_embeds, projector)
        target_embeds = encode_projector(target_embeds, projector)
    else:
        assert pred_target_embeds.dim() == 4, "Expected input shape (B, N, M, D)"   
        assert target_embeds.dim() == 4, "Expected input shape (B, N, M, D)"   
        pred_target_embeds = pred_target_embeds.mean(dim=2)  # Global average pooling over Mc
        target_embeds = target_embeds.mean(dim=2)  # Global average pooling over Mc
        
    Nc = pred_target_embeds.size(1)
    loss = 0.0
    
    # -- Compute VICReg loss for context embeddings. 
    pred_target_embeds_global = concat_all_gather_with_grad(pred_target_embeds)
    for i in range(Nc):
        for j in range(Nc):
            if i == j:
                continue
            else:
                z1_local = pred_target_embeds[:, i, :].contiguous()  # (B, D)
                z2_local = pred_target_embeds[:, j, :].contiguous()  # (B, D)
                z1_global = pred_target_embeds_global[:, i, :].contiguous()  # (B_total, D)
                z2_global = pred_target_embeds_global[:, j, :].contiguous()  # (B_total, D)
                loss += vicreg_loss(
                    z1_local,
                    z2_local,
                    z1_global,
                    z2_global,
                    sim_coeff=beta_sim,
                    var_coeff=beta_var,
                    cov_coeff=beta_cov,
                    eps=eps,
                    compute_invariance=False,
                )
                
    loss /= (Nc * (Nc - 1))
    
    # -- compute similarity loss between pred and target
    loss += F.mse_loss(pred_target_embeds, target_embeds) * beta_sim
    return loss

def cjepa_loss(
    target_embeds: torch.Tensor,
    pred_target_embeds: torch.Tensor,
    projector: nn.Module = None,
    beta_sim: float = 25,
    beta_var: float = 25,
    beta_cov: float = 1.0,
    eps: float = 1e-4,
):
    """[Fast Implementation] Compute CJepa loss between target embeddings and context embeddings.
    params: target_embeds (torch.Tensor): Target embeddings of shape (B, N, Mc, D) without predictor.
    params: pred_target_embeds (torch.Tensor): Predicted target embeddings of shape (B, N, Mc, D) from predictor.
    params: beta_sim (float): Coefficient for similarity loss.
    params: beta_var (float): Coefficient for variance loss.
    params: beta_cov (float): Coefficient for covariance loss.
    params: eps (float): Small constant for numerical stability.
    returns: torch.Tensor: Computed CJepa loss.
    """
    # --- Projection ---
    if projector is not None:
        pred_target_embeds = encode_projector(pred_target_embeds, projector)
        target_embeds = encode_projector(target_embeds, projector)
    else:
        pred_target_embeds = pred_target_embeds.mean(dim=2)
        target_embeds = target_embeds.mean(dim=2)
    
    Nc = pred_target_embeds.size(1)
    
    # 1. Gather global embeddings ONCE
    # (B_total, Nc, D)
    pred_target_embeds_global = concat_all_gather_with_grad(pred_target_embeds)
    
    # --- Optimization: Vectorized Regularization Calculation ---
    
    total_std_loss = 0.0
    total_cov_loss = 0.0
    
    for i in range(Nc):
        z_global = pred_target_embeds_global[:, i, :] # (B_total, D)
        total_std_loss += variance_loss(z_global, eps)
        total_cov_loss += convariance_loss(z_global)
    
    avg_std_loss = (2.0 / Nc) * total_std_loss
    avg_cov_loss = (2.0 / Nc) * total_cov_loss
    
    loss = beta_var * avg_std_loss + beta_cov * avg_cov_loss
    loss += F.mse_loss(pred_target_embeds, target_embeds) * beta_sim
    return loss
    
    
    
    
    
    
    
    