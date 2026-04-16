# smirk/models/backbones/inception_resnetv1.py - InceptionResnetV1 model backbones

# registers the InceptionResnetV1 model backbones and defines model loader
# inception_resnetv1_vggface2 - pretrained on VGGFace2
# inception_resnetv1_casia    - pretrained on CASIA-WebFace
 
import torch.nn as nn
from smirk.models.registry import register_model, get_weights
from smirk.models.stats import ALL_MEANS, ALL_STDS
from facenet_pytorch import InceptionResnetV1
from smirk.models.definitions import inceptionresnetv1_4finetune
from smirk.utils.files import get_path


def load_inception_resnetv1_vggface2_E(spec, num_experts, num_classification, device, load_weights):
    model = inceptionresnetv1_4finetune.InceptionResnetV1_4finetune_E(
        classify=True,
        pretrained="vggface2",
        num_classes=num_classification,
        num_experts=num_experts
    )
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            model.load_state_dict(state_dict)

    # replace layers
    model.logits = nn.ModuleList(
        [nn.Linear(512, num_classification) for _ in range(num_experts)]
    )
    model.last_linear = nn.ModuleList(
        [nn.Linear(1792, 512, bias=False) for _ in range(num_experts)]
    )

    # selectively enable gradients
    for param in model.parameters():
        param.requires_grad = False
    for param in model.last_linear.parameters():
        param.requires_grad = True
    for param in model.logits.parameters():
        param.requires_grad = True

    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.last_linear.parameters(),
        model.logits.parameters()
    ]

    model = model.to(device)
    return model


def load_inception_resnetv1_casia_E(spec, num_experts, num_classification, device, load_weights):
    model = inceptionresnetv1_4finetune.InceptionResnetV1_4finetune_E(
        classify=True,
        pretrained="casia-webface",
        num_classes=num_classification,
        num_experts=num_experts
    )
    
    if load_weights:
        state_dict = get_weights(spec, device)

        if state_dict:
            state_dict = {k: v for k, v in state_dict.items() if 'logits' not in k}
            model.load_state_dict(state_dict, strict=False)

    # replace layers
    model.logits = nn.ModuleList(
        [nn.Linear(512, num_classification) for _ in range(num_experts)]
    )
    model.last_linear = nn.ModuleList(
        [nn.Linear(1792, 512, bias=False) for _ in range(num_experts)]
    )

    # selectively enable gradients
    for param in model.parameters():
        param.requires_grad = False
    for param in model.last_bn.parameters():
        param.requires_grad = True
    for param in model.last_linear.parameters():
        param.requires_grad = True
    for param in model.logits.parameters():
        param.requires_grad = True

    # store parameters for modified layers (for building surrogate optimizer)
    model.modified_layer_parameters = [
        model.last_bn.parameters(),
        model.last_linear.parameters(),
        model.logits.parameters()
    ]

    model = model.to(device)
    return model


@register_model(
    "inception_resnetv1_vggface2",
    resolution = 160,
    mean = ALL_MEANS["inception_resnetv1_vggface2"],
    std = ALL_STDS["inception_resnetv1_vggface2"],
    expert_wrapper=load_inception_resnetv1_vggface2_E,
    weights_path=get_path("smirk/models/weights/20180402-114759-vggface2.pt")
)
def load_inception_resnetv1_vggface2():
    model = InceptionResnetV1(classify=True, pretrained='vggface2')

    return model

@register_model(
    "inception_resnetv1_casia",
    resolution = 160,
    mean = ALL_MEANS["inception_resnetv1_casia"],
    std = ALL_STDS["inception_resnetv1_casia"],
    expert_wrapper=load_inception_resnetv1_casia_E,
    weights_path=get_path("smirk/models/weights/20180408-102900-casia-webface.pt")
)
def load_inception_resnetv1_casia():
    model = InceptionResnetV1(classify=True, pretrained='casia-webface')

    return model

