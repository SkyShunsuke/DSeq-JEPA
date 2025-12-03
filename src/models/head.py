
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