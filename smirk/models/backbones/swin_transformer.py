# smirk/models/backbones/swin_transformer.py - Swin Transformer model backbone

# registers the Swin Transformer V2 Tiny model backbone and defines model loader

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2
from torchvision.ops.misc import MLP

from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import swintransformer_4finetune
from smirk.utils.files import get_path


def load_swin_transformer_E(spec, num_experts, num_classification, device, load_weights):
    model = swintransformer_4finetune.SwinTransformer_E(
        num_classes=8631, 
        num_experts=num_experts,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model_pretrain = models.swin_transformer.swin_v2_t(pretrained=True)
            model_pretrain.head = nn.Linear(768, 8631)
            model_pretrain.load_state_dict(state_dict)
            model.load_state_dict(model_pretrain.state_dict())
    
    # modify layers
    model.head = nn.ModuleList(
        [nn.Linear(768, num_classification) for _ in range(num_experts)]
    )
    model.features[-1][-1].mlp = nn.ModuleList(
        [MLP(768, [3072, 768], activation_layer=nn.GELU, dropout=0.0) for _ in range(num_experts)]
    )
    
    # selectively enable gradients for new layers only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.features[-1][-1].mlp.parameters():
        param.requires_grad = True
    
    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.features[-1][-1].mlp.parameters(),
        model.head.parameters()
    ]
    
    model = model.to(device)
    return model


@register_model(
    "swin_transformer",
    resolution=260,
    mean=ALL_MEANS["swin_transformer"],
    std=ALL_STDS["swin_transformer"],
    expert_wrapper=load_swin_transformer_E
)
def load_swin_transformer():
    model = models.swin_transformer.swin_v2_t(pretrained=True)
    model.head = nn.Linear(768, 8631)

    return model
