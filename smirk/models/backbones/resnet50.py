# smirk/models/backbones/resnet50.py - Resnet50 model backbone

# registers the resnet50 model backbone and defines model loader

import torch
import torch.nn as nn
 
from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import resnet50_scratch_dag as resnet50
from smirk.utils.files import get_path

def load_resnet50_E(spec, num_experts, device):
    model = resnet50.Resnet50_scratch_dag_E(num_experts)

    return model

@register_model(
    "resnet50",
    resolution = 224,
    mean = ALL_MEANS["resnet50"],
    std = ALL_STDS["resnet50"],
    expert_wrapper=load_resnet50_E,
    weights_path=get_path("smirk/models/weights/resnet50_scratch_dag.pth")
)
def load_resnet50(spec, device):
    model = resnet50.Resnet50_scratch_dag()

    return model

