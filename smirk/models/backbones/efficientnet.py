# smirk/models/backbones/efficientnet.py - EfficientNet model backbones

# registers the EfficientNet model backbones and defines model loaders
# efficientnet_b0       - EfficientNetB0 (8631 classes)
# efficientnet_b0_casia - EfficientNetB0 trained on CASIA-WebFace (10575 classes)

import torch
import torch.nn as nn
import torchvision.models as models

from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.utils.files import get_path


@register_model(
    "efficientnet_b0",
    resolution=256,
    mean=ALL_MEANS["efficientnet_b0"],
    std=ALL_STDS["efficientnet_b0"]
)
def load_efficientnet_b0(spec, device):
    model = models.efficientnet.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 8631),
    )

    return model


@register_model(
    "efficientnet_b0_casia",
    resolution=256,
    mean=ALL_MEANS["efficientnet_b0_casia"],
    std=ALL_STDS["efficientnet_b0_casia"],
    #weights_path=get_path("smirk/models/weights/efficientnet_b0_best_model_casia.pth")
)
def load_efficientnet_b0_casia(spec, device):
    model = models.efficientnet.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 10575),
    )

    return model
