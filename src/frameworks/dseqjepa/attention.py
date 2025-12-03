import torch
import torch.nn as nn
import numpy as np
import cv2

class SelfGuidedAttention(nn.Module):
    """Self-Guided Attention module as used in DSeq-JEPA. We simply call it as 'Attention' in the paper.
    """
    def __init__(self, strategy, depth, alpha, pool_window_size, region_num):
        super().__init__()
        self.strategy = strategy
        self.depth = depth
        self.alpha = alpha  # Minimum area threshold ratio (e.g., 0.01)
        self.pool_window_size = pool_window_size
        self.region_num = region_num # N in the paper (e.g., 5)

    def forward(self, z, query_z):
        """
        params:
            z: torch.Tensor: Input feature map of shape (B, N, D), where N = h * w
            query_z: torch.Tensor: Query token of shape (B, 1, D)
        returns:
            regions: torch.Tensor: Selected region mask of shape (B, region_num, N)
        """
        if self.strategy == 'similarity':
            B, N, D = z.shape
            # Assuming square image for reshaping (N = H * W)
            H = W = int(N ** 0.5) 
            
            # Compute similarity between query and all tokens (Saliency Map A)
            sim = torch.matmul(query_z, z.transpose(1, 2)) / (D ** 0.5)  # (B, 1, N)
            sim = sim.squeeze(1)  # (B, N) -> Saliency Map A

            # 1. Normalize the saliency map (Eq. 1)
            # A_tilde = (A - min(A)) / (max(A) - min(A))
            sim_min = sim.amin(dim=1, keepdim=True)
            sim_max = sim.amax(dim=1, keepdim=True)
            norm_sim = (sim - sim_min) / (sim_max - sim_min + 1e-6) # (B, N)
            
            norm_sim_img = norm_sim.view(B, H, W)

            batch_masks = []

            norm_sim_np = norm_sim_img.detach().cpu().numpy()

            for b in range(B):
                saliency = norm_sim_np[b]
                saliency_uint8 = (saliency * 255).astype(np.uint8)

                # 2. Otsu's Method (Eq. 2)
                thresh_val, binary_mask = cv2.threshold(
                    saliency_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # 3. Connected-Component Labeling with 8-neighborhood
                num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)

                candidate_regions = []
                
                for l in range(1, num_labels):
                    mask = (labels == l)
                    area = mask.sum()

                    # Small components filter (|R_k| < alpha * hw)
                    if area < (self.alpha * N):
                        continue

                    # Discriminative score rho_k (Eq. 3)
                    score = saliency[mask].mean()
                    candidate_regions.append((score, mask))

                # 4. Ranking and Selection
                candidate_regions.sort(key=lambda x: x[0], reverse=True)

                # Top-(N-1) regions
                top_k_count = self.region_num - 1
                selected_regions = candidate_regions[:top_k_count]

                sample_region_masks = np.zeros((self.region_num, N), dtype=bool)
                
                union_mask = np.zeros((H, W), dtype=bool)

                for i, (score, mask_2d) in enumerate(selected_regions):
                    sample_region_masks[i, :] = mask_2d.flatten()
                    union_mask = np.logical_or(union_mask, mask_2d)

                # 5. Last Region R_N (Background)
                # R_N = Omega \ Union(R_k) 
                background_mask = ~union_mask
                sample_region_masks[self.region_num - 1, :] = background_mask.flatten()

                batch_masks.append(torch.from_numpy(sample_region_masks))

            regions = torch.stack(batch_masks).to(device=z.device, dtype=z.dtype) 
            regions = regions.float()

            return regions

        else:
            raise NotImplementedError(f"Strategy {self.strategy} is not implemented.")