# smirk/models/backbones/inception_v3.py - Inception-V3 model backbone

# registers the Inception-V3 model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.utils.files import get_path


class InceptionAux(nn.Module):
    """Auxiliary classifier for Inception-V3"""
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 768, kernel_size=5),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv0(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@register_model(
    "inception_v3",
    resolution=342,
    mean=ALL_MEANS["inception_v3"],
    std=ALL_STDS["inception_v3"],
    #weights_path=get_path("smirk/models/weights/inception_v3_best_model.pth")
)
def load_inception_v3():
    model = models.inception_v3(pretrained=True)
    
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits = InceptionAux(768, 8631)
        model.fc = nn.Linear(2048, 8631)

    return model
