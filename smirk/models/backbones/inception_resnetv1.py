# smirk/models/backbones/inception_resnetv1.py - InceptionResnetV1 model backbones

# registers the InceptionResnetV1 model backbones and defines model loader
# inception_resnetv1_vggface2 - pretrained on VGGFace2
# inception_resnetv1_casia    - pretrained on CASIA-WebFace
 
from smirk.models.registry import register_model
from smirk.models.stats import ALL_MEANS, ALL_STDS
from facenet_pytorch import InceptionResnetV1
from smirk.utils.files import get_path


@register_model(
    "inception_resnetv1_vggface2",
    resolution = 160,
    mean = ALL_MEANS["inception_resnetv1_vggface2"],
    std = ALL_STDS["inception_resnetv1_vggface2"]
)
def load_inception_resnetv1_vggface2():
    model = InceptionResnetV1(classify=True, pretrained='vggface2')

    return model

@register_model(
    "inception_resnetv1_casia",
    resolution = 160,
    mean = ALL_MEANS["inception_resnetv1_casia"],
    std = ALL_STDS["inception_resnetv1_casia"]
)
def load_inception_resnetv1_casia():
    model = InceptionResnetV1(classify=True, pretrained='casia-webface')

    return model

