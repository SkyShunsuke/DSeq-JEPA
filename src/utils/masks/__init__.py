from src.utils.masks.multiblock import MaskCollator as MultiBlockMaskCollator
from src.utils.masks.random import MaskCollator as RandomMaskCollator

def get_mask_collator(mask_type, **kwargs):
    if mask_type == 'multiblock':
        return MultiBlockMaskCollator(**kwargs)
    elif mask_type == 'random':
        return RandomMaskCollator(**kwargs)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
