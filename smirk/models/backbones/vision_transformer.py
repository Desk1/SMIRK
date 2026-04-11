# smirk/models/backbones/vision_transformer.py - Vision Transformer model backbone

# registers the Vision Transformer model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import vitb16_4finetune
from smirk.utils.files import get_path


def load_vision_transformer_E(spec, num_experts, num_classification, device, load_weights):
    model = vitb16_4finetune.VisionTransformer_E(
        num_classes=8631, 
        num_experts=num_experts, 
        patch_size=16, 
        num_layers=12, 
        num_heads=12, 
        hidden_dim=768, 
        mlp_dim=3072, 
        image_size=224
    )
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model_pretrain = models.vision_transformer.vit_b_16(pretrained=True)
            heads_layers1: OrderedDict[str, nn.Module] = OrderedDict()
            heads_layers1["head"] = nn.Linear(768, 8631)
            model_pretrain.heads = nn.Sequential(heads_layers1)
            model_pretrain.load_state_dict(state_dict)
            model.load_state_dict(model_pretrain.state_dict())
    
    # modify layers
    model.heads = nn.ModuleList(
        [nn.Linear(768, num_classification) for _ in range(num_experts)]
    )
    
    # selectively enable gradients
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True
    for param in model.encoder.layers[-1].mlp.parameters():
        param.requires_grad = True
    
    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.encoder.layers[-1].mlp.parameters(),
        model.heads.parameters()
    ]
    
    model = model.to(device)
    return model


@register_model(
    "vision_transformer",
    resolution=224,
    mean=ALL_MEANS["vision_transformer"],
    std=ALL_STDS["vision_transformer"],
    expert_wrapper=load_vision_transformer_E
)
def load_vision_transformer():
    model = models.vision_transformer.vit_b_16(pretrained=True)
    
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    heads_layers["head"] = nn.Linear(768, 8631)
    model.heads = nn.Sequential(heads_layers)

    return model
