# smirk/models/backbones/inception_resnetv1.py - InceptionResnetV1 model backbones

# registers the InceptionResnetV1 model backbones and defines model loader
# inception_resnetv1_vggface2 - pretrained on VGGFace2
# inception_resnetv1_casia    - pretrained on CASIA-WebFace

import torch
import torch.nn as nn
 
from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from smirk.models.definitions import net_sphere
from smirk.utils.files import get_path


@register_model(
    "sphere20a",
    resolution = (112, 96),
    mean = ALL_MEANS["sphere20a"],
    std = ALL_STDS["sphere20a"],
    weights_path=get_path("smirk/models/weights/sphere20a_20171020.pth")
)
def load_sphere20a():
    model = net_sphere.sphere20a()

    return model


