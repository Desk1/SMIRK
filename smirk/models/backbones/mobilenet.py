# smirk/models/backbones/mobilenet.py - MobileNet model backbone

# registers the MobileNetV2 model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models

from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.utils.files import get_path


@register_model(
    "mobilenet_v2",
    resolution=224,
    mean=ALL_MEANS["mobilenet_v2"],
    std=ALL_STDS["mobilenet_v2"],
    #weights_path=get_path("smirk/models/weights/mobilenet_v2_best_model.pth")
)
def load_mobilenet_v2(spec, device):
    model = models.mobilenet.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 8631),
    )

    return model
