# smirk/models/backbones/resnet50.py - Resnet50 model backbone

# registers the resnet50 model backbone and defines model loader

import torch.nn as nn
 
from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import resnet50_scratch_dag as resnet50
from smirk.utils.files import get_path

def load_resnet50_E(spec, num_experts, num_classification, device, load_weights):
    model = resnet50.Resnet50_scratch_dag_E(num_experts)
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            # Exclude per-expert layers from loaded state_dict
            state_dict = {k: v for k, v in state_dict.items() if 'conv5_3_1x1_increase' not in k and 'classifier' not in k}
            model.load_state_dict(state_dict, strict=False)

    # replace layers with per-expert versions to prevent BatchNorm entanglement
    model.classifier = nn.ModuleList(
        [nn.Conv2d(2048, num_classification, kernel_size=[1, 1], stride=(1, 1)) for _ in range(num_experts)]
    )
    model.conv5_3_1x1_increase = nn.ModuleList(
        [nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False) for _ in range(num_experts)]
    )
    # Make BatchNorm per-expert as well to prevent entanglement
    model.conv5_3_1x1_increase_bn = nn.ModuleList(
        [nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for _ in range(num_experts)]
    )

    # selectively enable gradients
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv5_3_1x1_increase.parameters():
        param.requires_grad = True
    for param in model.conv5_3_1x1_increase_bn.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.conv5_3_1x1_increase.parameters(),
        model.conv5_3_1x1_increase_bn.parameters(),
        model.classifier.parameters()
    ]

    model = model.to(device)
    return model

@register_model(
    "resnet50",
    resolution = 224,
    mean = ALL_MEANS["resnet50"],
    std = ALL_STDS["resnet50"],
    expert_wrapper=load_resnet50_E,
    weights_path=get_path("smirk/models/weights/resnet50_scratch_dag.pth")
)
def load_resnet50():
    model = resnet50.Resnet50_scratch_dag()

    return model

