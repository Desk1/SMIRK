# smirk/models/backbones/vgg.py - VGG model backbones

# registers the VGG model backbones and defines model loaders
# vgg16bn - VGG-16 with batch normalization
# vgg16   - VGG-16 Face model

import torch
import torch.nn as nn
import torchvision.models as models

from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import vgg_m_face_bn_dag as vggbn
from smirk.models.definitions import vgg_face_dag as vgg
from smirk.utils.files import get_path


@register_model(
    "vgg16bn",
    resolution=224,
    mean=ALL_MEANS["vgg16bn"],
    std=ALL_STDS["vgg16bn"],
    weights_path=get_path("smirk/models/weights/vgg_m_face_bn_dag.pth")
)
def load_vgg16bn():
    model = vggbn.Vgg_m_face_bn_dag()
    
    return model


@register_model(
    "vgg16",
    resolution=224,
    mean=ALL_MEANS["vgg16"],
    std=ALL_STDS["vgg16"],
    weights_path=get_path("smirk/models/weights/vgg_face_dag.pth")
)
def load_vgg16():
    model = vgg.Vgg_face_dag()

    return model
