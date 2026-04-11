# smirk/models/backbones/efficientnet.py - EfficientNet model backbones

# registers the EfficientNet model backbones and defines model loaders
# efficientnet_b0       - EfficientNetB0 (8631 classes)
# efficientnet_b0_casia - EfficientNetB0 trained on CASIA-WebFace (10575 classes)

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import _efficientnet_conf
from torchvision.ops.misc import Conv2dNormActivation

from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import efficientnetb0_4finetune


def load_efficientnet_b0_E(spec, num_experts, num_classification, device, load_weights):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    model = efficientnetb0_4finetune.EfficientNet_E(
        inverted_residual_setting, 
        num_classes=8631, 
        num_experts=num_experts, 
        dropout=0.2
    )
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model_pretrain = models.efficientnet.efficientnet_b0(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 8631),
            )
            model_pretrain.load_state_dict(state_dict)
            model.load_state_dict(model_pretrain.state_dict())
    
    model.classifier = nn.ModuleList([nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classification),
    ) for _ in range(num_experts)])
    
    # Remove last layer and add expert specific layers
    new_features = list(model.features.children())[:-1]
    model.features = nn.Sequential(*new_features)
    for _ in range(num_experts):
        model.features.append(Conv2dNormActivation(
            320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU
        ))
    
    # selectively enable gradients=
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features[-1*num_experts:].parameters():
        param.requires_grad = True
    
    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.features[-1*num_experts:].parameters(),
        model.classifier.parameters()
    ]
    
    model = model.to(device)
    return model


def load_efficientnet_b0_casia_E(spec, num_experts, num_classification, device, load_weights):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    model = efficientnetb0_4finetune.EfficientNet_E(
        inverted_residual_setting, 
        num_classes=10575, 
        num_experts=num_experts, 
        dropout=0.2
    )
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model_pretrain = models.efficientnet.efficientnet_b0(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 8631),
            )
            model_pretrain.load_state_dict(state_dict)
            model.load_state_dict(model_pretrain.state_dict())
    
    model.classifier = nn.ModuleList([nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classification),
    ) for _ in range(num_experts)])
    
    # Remove last layer and add expert-specific layers
    new_features = list(model.features.children())[:-1]
    model.features = nn.Sequential(*new_features)
    for _ in range(num_experts):
        model.features.append(Conv2dNormActivation(
            320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU
        ))
    
    # selectively enable gradients
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features[-1*num_experts:].parameters():
        param.requires_grad = True
    
    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.features[-1*num_experts:].parameters(),
        model.classifier.parameters()
    ]
    
    model = model.to(device)
    return model


@register_model(
    "efficientnet_b0",
    resolution=256,
    mean=ALL_MEANS["efficientnet_b0"],
    std=ALL_STDS["efficientnet_b0"],
    expert_wrapper=load_efficientnet_b0_E
)
def load_efficientnet_b0():
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
    expert_wrapper=load_efficientnet_b0_casia_E
)
def load_efficientnet_b0_casia():
    model = models.efficientnet.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 10575),
    )

    return model
