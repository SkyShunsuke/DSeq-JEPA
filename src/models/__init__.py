from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor, init_weights
from src.models.masked_vision_transformer import MaskedVisionTransformer, MaskedVisionTransformerPredictor
from src.models.head import LinearEvalModel, FrozenFeatureClassifier
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
    use_class_token: bool = False,
    drop_path_rate: float = 0.0,
    **kwargs,
):
    vit_model = vit if not use_masked_vit else masked_vit
    target_encoder = vit_model.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_projector=False,
        drop_path_rate=drop_path_rate,
        use_class_token=use_class_token,
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

def init_low_level_model(
    backbone: torch.nn.Module,
    freeze_backbone: bool = True,
    embed_dim: int = 768,
    num_classes: int = 8,
    feature_key: str = 'lastCLS',
    head_type: str = 'linear',
    use_bn: bool = False,
    mlp_config: dict = None,
    **kwargs,
):
    """Frozen encoder + task-specific linear head (Clevr/Count, Clevr/Dist)."""
    return FrozenFeatureClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        feature_key=feature_key,
        freeze_backbone=freeze_backbone,
        head_type=head_type,
        use_bn=use_bn,
        mlp_config=mlp_config,
    )

def init_detection_model(
    encoder: torch.nn.Module,
    num_classes: int = 91,
    out_layers=(3, 5, 7, 11),
    fpn_channels: int = 256,
    img_size: int = 1024,
    window_size: int = 14,
    use_checkpoint: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
):
    """Mask R-CNN (FPN) on top of the pre-trained encoder -- MS-COCO."""
    from src.models.detection import build_mask_rcnn
    return build_mask_rcnn(
        encoder,
        num_classes=num_classes,
        out_layers=tuple(out_layers),
        fpn_channels=fpn_channels,
        img_size=img_size,
        window_size=window_size,
        use_checkpoint=use_checkpoint,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )

def init_segmentation_model(
    encoder: torch.nn.Module,
    num_classes: int = 150,
    out_layers=(3, 5, 7, 11),
    channels: int = 512,
    pool_scales=(1, 2, 3, 6),
    dropout: float = 0.1,
    use_aux_head: bool = True,
    aux_index: int = 2,
    aux_channels: int = 256,
    window_size: int = 0,
    use_checkpoint: bool = False,
    freeze_backbone: bool = False,
    **kwargs,
):
    """UPerNet on top of the pre-trained encoder -- ADE20K."""
    from src.models.vit_adapter import ViTDenseBackbone
    from src.models.upernet import ViTUPerNet
    backbone = ViTDenseBackbone(
        encoder,
        out_layers=tuple(out_layers),
        window_size=window_size,
        use_checkpoint=use_checkpoint,
        freeze=freeze_backbone,
    )
    return ViTUPerNet(
        backbone,
        num_classes=num_classes,
        channels=channels,
        pool_scales=tuple(pool_scales),
        dropout=dropout,
        use_aux_head=use_aux_head,
        aux_index=aux_index,
        aux_channels=aux_channels,
    )
