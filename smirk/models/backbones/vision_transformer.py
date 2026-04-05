# smirk/models/backbones/vision_transformer.py - Vision Transformer model backbone

# registers the Vision Transformer model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.utils.files import get_path


@register_model(
    "vision_transformer",
    resolution=224,
    mean=ALL_MEANS["vision_transformer"],
    std=ALL_STDS["vision_transformer"],
    weights_path=get_path("smirk/models/weights/vision_transformer_2_best_model.pth")
)
def load_vision_transformer(spec, device):
    model = models.vision_transformer.vit_b_16(pretrained=True)
    
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    heads_layers["head"] = nn.Linear(768, 8631)
    model.heads = nn.Sequential(heads_layers)

    model = model.to(device)
    return model
