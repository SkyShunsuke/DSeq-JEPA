from torch.amp import GradScaler

def get_gradient_scaler(
    use_bf16: bool = True,
    device: str = "cuda",
) -> GradScaler:
    """
    Returns a GradScaler for mixed precision training.
    """
    return GradScaler(device=device, enabled=use_bf16)