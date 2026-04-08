# smirk/models/backbones/swin_transformer.py - Swin Transformer model backbone

# registers the Swin Transformer V2 Tiny model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models

from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.utils.files import get_path


@register_model(
    "swin_transformer",
    resolution=260,
    mean=ALL_MEANS["swin_transformer"],
    std=ALL_STDS["swin_transformer"]
)
def load_swin_transformer():
    model = models.swin_transformer.swin_v2_t(pretrained=True)
    model.head = nn.Linear(768, 8631)

    return model
