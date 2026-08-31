
import torch
import torch.nn as nn

MLP_CONFIG = {
    'hidden_dim': 2048,
    'num_layers': 2,
    'use_bn': True,
    'activation': nn.ReLU,
}

class PoolerHead(nn.Module):
    """GAP => Concat => BN => Linear classifier
    params: in_dim: int, input feature dimension    
    num_classes: int, number of classes for classification
    
    Note: expected input shape: (B, in_dim)
    """
    def __init__(self, in_dim, num_classes, head_type='linear', use_bn=True, mlp_config=None, **kwargs):
        super().__init__()
        self.channel_bn = nn.BatchNorm2d(in_dim) if use_bn else nn.Identity()
        
        if head_type == "linear":
            self.out_proj = nn.Linear(in_dim, num_classes)
        elif head_type == "mlp":
            config = mlp_config if mlp_config is not None else MLP_CONFIG
            layers = []
            input_dim = in_dim
            for i in range(config['num_layers'] - 1):
                layers.append(nn.Linear(input_dim, config['hidden_dim']))
                if config['use_bn']:
                    layers.append(nn.BatchNorm1d(config['hidden_dim']))
                layers.append(config['activation']())
                input_dim = config['hidden_dim']
            layers.append(nn.Linear(input_dim, num_classes))
            self.out_proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, batch: torch.Tensor):
        batch = batch.unsqueeze(2).unsqueeze(3)
        out = self.channel_bn(batch)
        out = torch.flatten(out, start_dim=1)
        out = self.out_proj(out)
        return out
    
class LinearEvalModel(nn.Module):
    """Frozen backbone + trainable heads"""
    def __init__(self, backbone, freeze_backbone=True, embed_dim=768, num_classes=1000, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.concat_pool4 = PoolerHead(in_dim=embed_dim * 4, num_classes=num_classes, **kwargs)
        self.last_pool = PoolerHead(in_dim=embed_dim, num_classes=num_classes, **kwargs)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self 

    def forward(self, x, **kwargs):
        with torch.no_grad():
            feats = self.backbone.get_intermediate_features(x, ["lastPOOL", "concatPOOL4"])
        last_feat = feats["lastPOOL"]
        concat_feat = feats["concatPOOL4"]
        out_pool4 = self.concat_pool4(concat_feat)
        out_last = self.last_pool(last_feat)
        return out_pool4, out_last

def feature_dim_of(feature_key: str, embed_dim: int) -> int:
    """Width of the representation produced by `VisionTransformer.get_intermediate_features`."""
    for prefix in ("concatCLS", "concatPOOL", "concatBLK"):
        if feature_key.startswith(prefix):
            return embed_dim * int(feature_key.replace(prefix, ""))
    if feature_key.startswith("stridePOOL_"):
        parts = [int(s) for s in feature_key.replace("stridePOOL_", "").split("_")]
        return embed_dim * parts[0]
    return embed_dim


class FrozenFeatureClassifier(nn.Module):
    """Frozen backbone + a single classification head on one chosen feature.

    Used for the low-level reasoning transfer (Clevr/Count, Clevr/Dist), where
    the paper freezes the target encoder and trains a task-specific linear
    classifier on the final-layer [CLS] token (`feature_key='lastCLS'`).
    """

    def __init__(
        self,
        backbone,
        embed_dim: int = 768,
        num_classes: int = 8,
        feature_key: str = "lastCLS",
        freeze_backbone: bool = True,
        head_type: str = "linear",
        use_bn: bool = False,
        mlp_config: dict = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_key = feature_key
        self.head = PoolerHead(
            in_dim=feature_dim_of(feature_key, embed_dim),
            num_classes=num_classes,
            head_type=head_type,
            use_bn=use_bn,
            mlp_config=mlp_config,
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, **kwargs):
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_features(x, [self.feature_key])
        else:
            feats = self.backbone.get_intermediate_features(x, [self.feature_key])
        return self.head(feats[self.feature_key])
