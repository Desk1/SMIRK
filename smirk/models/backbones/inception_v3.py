# smirk/models/backbones/inception_v3.py - Inception-V3 model backbone

# registers the Inception-V3 model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import InceptionE

from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import inceptionv3_4finetune


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


def load_inception_v3_E(spec, num_experts, num_classification, device, load_weights):
    model = inceptionv3_4finetune.Inception3_E(num_classes=8631, num_experts=num_experts)
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model_pretrain = models.inception_v3(pretrained=True)
            if hasattr(model_pretrain, 'AuxLogits'):
                model_pretrain.AuxLogits = InceptionAux(768, 8631)  
                model_pretrain.fc = nn.Linear(2048, 8631)
            model_pretrain.load_state_dict(state_dict)
            model.load_state_dict(model_pretrain.state_dict())
    
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits = nn.ModuleList(
            [InceptionAux(768, num_classification) for _ in range(num_experts)]
        )
        model.fc = nn.ModuleList(
            [nn.Linear(2048, num_classification) for _ in range(num_experts)]
        )
    
    model.Mixed_7c = nn.ModuleList(
        [InceptionE(2048) for _ in range(num_experts)]
    )
    
    # selectively enable gradients
    for param in model.parameters():
        param.requires_grad = False
    for param in model.AuxLogits.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.Mixed_7c.parameters():
        param.requires_grad = True
    
    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.Mixed_7c.parameters(),
        model.AuxLogits.parameters(),
        model.fc.parameters()
    ]
    
    model = model.to(device)
    return model


@register_model(
    "inception_v3",
    resolution=342,
    mean=ALL_MEANS["inception_v3"],
    std=ALL_STDS["inception_v3"],
    expert_wrapper=load_inception_v3_E
)
def load_inception_v3():
    model = models.inception_v3(pretrained=True)
    
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits = InceptionAux(768, 8631)
        model.fc = nn.Linear(2048, 8631)

    return model
