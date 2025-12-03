from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor, init_weights
from src.models.masked_vision_transformer import MaskedVisionTransformer, MaskedVisionTransformerPredictor
from src.models.head import LinearEvalModel
from src.models import vision_transformer as vit
from src.models import masked_vision_transformer as masked_vit
from src.models.projector import VICRegProjector

import torch

def init_model(
    device, 
    patch_size: int = 16,
    use_projector: bool = False,
    model_name: str = 'vit_base',
    crop_size: int = 224,
    pred_depth: int = 6,
    emb_dim: int = 768,
    pred_emb_dim: int = 384,
    include_mask_token: bool = True,
    learned_pos_emb: bool = False,
    apply_stop: bool = False,
    drop_path_rate: float = 0.0,
    stop_var: float = 0.0,
    use_class_token: bool = False,
    use_masked_vit: bool = False,
    **kwargs,
):
    vit_model = vit if not use_masked_vit else masked_vit
    
    encoder = vit_model.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_projector=use_projector,
        drop_path_rate=drop_path_rate,
        use_class_token=use_class_token,
    ).to(device)
    
    predictor = vit_model.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        emb_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        include_mask_token=include_mask_token,
        learned_pos_emb=learned_pos_emb,
        apply_stop=apply_stop,
        stop_var=stop_var,
    ).to(device)
    
    for m in encoder.modules():
        init_weights(m)
    
    for m in predictor.modules():
        init_weights(m)
        
    return encoder, predictor

def init_target_encoder(
    device, 
    patch_size: int = 16,
    model_name: str = 'vit_base',
    crop_size: int = 224,
    use_masked_vit: bool = False,
    **kwargs,
):
    vit_model = vit if not use_masked_vit else masked_vit
    target_encoder = vit_model.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_projector=False,
        drop_path_rate=0.0,
    ).to(device)
    for m in target_encoder.modules():
        init_weights(m)
    return target_encoder

def init_probing_model(
    backbone: torch.nn.Module,
    freeze_backbone: bool = True,
    embed_dim: int = 768,
    num_classes: int = 1000,
    head_type: str = 'linear',
    use_bn: bool = True,
    mlp_config: dict = None,
    **kwargs,
):
    head = LinearEvalModel(
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        head_type=head_type,
        use_bn=use_bn,
        mlp_config=mlp_config,
        **kwargs,
    )
    return head

def init_projector(
    projector_type: str = 'vicreg',
    in_dim: int = 768,
    hidden_dim: int = 3072,
    out_dim: int = 3072,
    num_layers: int = 3,
    fc_bias: bool = False,
    **kwargs,
):
    if projector_type == 'vicreg':
        projector = VICRegProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            fc_bias=fc_bias,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported projector_type: {projector_type}")
    return projector