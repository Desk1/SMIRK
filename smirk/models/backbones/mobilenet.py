# smirk/models/backbones/mobilenet.py - MobileNet model backbone

# registers the MobileNetV2 model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops.misc import Conv2dNormActivation

from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import mobilenetv2_4finetune


def load_mobilenet_v2_E(spec, num_experts, num_classification, device, load_weights):
    model = mobilenetv2_4finetune.MobileNetV2_E(num_classes=8631, num_experts=num_experts)
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model_pretrain = models.mobilenet.mobilenet_v2(pretrained=True)
            model_pretrain.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model_pretrain.last_channel, 8631),
            )
            model_pretrain.load_state_dict(state_dict)
            model.load_state_dict(model_pretrain.state_dict())
    
    model.classifier = nn.ModuleList([
        nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classification),
        ) for _ in range(num_experts)
    ])
    
    # Remove last layer and add expert specific layers
    new_features = list(model.features.children())[:-1]
    model.features = nn.Sequential(*new_features)
    for _ in range(num_experts):
        model.features.append(Conv2dNormActivation(
            320, 1280, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6
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
    "mobilenet_v2",
    resolution=224,
    mean=ALL_MEANS["mobilenet_v2"],
    std=ALL_STDS["mobilenet_v2"],
    expert_wrapper=load_mobilenet_v2_E
)
def load_mobilenet_v2():
    model = models.mobilenet.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 8631),
    )

    return model
