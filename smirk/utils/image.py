# smirk/utils/image.py - image processing utils

"""
API
-----------------------------------------
normalize(image_tensor, arch_name)        # normalise [0,255] RGB  model input
denormalize(image_tensor, arch_name)      # invert normalisationRGB
resize_img(img, image_resolution)         # resize img
crop_img(img, arch_name)                  # centre crop based on architecture
crop_and_resize(img, arch_name, res)      # crop and resize img
"""

import torch
import torchvision.transforms.functional as F

#############################################
# Architecture specific normalisation stats #
#############################################

ALL_MEANS = {
    # VGG-Face
    "vgg16":                            [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_5class":                     [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_8class":                     [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_9class":                     [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_16class":                    [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_24class":                    [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_10class_dp_sgd":             [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16_10class":                    [129.186279296875, 104.76238250732422, 93.59396362304688],
    "vgg16bn":                          [131.45376586914062, 103.98748016357422, 91.46234893798828],
    # VGG-Face 2 / ResNet-50
    "resnet50":                         [131.0912, 103.8827, 91.4953],
    "mobilenet_v2":                     [131.0912, 103.8827, 91.4953],
    "efficientnet_b0":                  [131.0912, 103.8827, 91.4953],
    "inception_v3":                     [131.0912, 103.8827, 91.4953],
    "swin_transformer":                 [131.0912, 103.8827, 91.4953],
    "vision_transformer":               [131.0912, 103.8827, 91.4953],
    # FaceNet / SphereFace
    "inception_resnetv1_vggface2":      [127.5, 127.5, 127.5],
    "inception_resnetv1_vggface2_8631": [127.5, 127.5, 127.5],
    "inception_resnetv1_casia":         [127.5, 127.5, 127.5],
    "sphere20a":                        [127.5, 127.5, 127.5],
    "efficientnet_b0_casia":            [127.5, 127.5, 127.5],
    # CCS-19
    "ccs19ami_facescrub":               [0.0, 0.0, 0.0],
    "ccs19ami_facescrub_rgb":           [0.0, 0.0, 0.0],
    # Misc
    "azure":                            [0.0, 0.0, 0.0],
    "cat_resnet18":                     [0.485 * 255, 0.456 * 255, 0.406 * 255],
    "resnet18_10class":                 [0.485 * 255, 0.456 * 255, 0.406 * 255],
    "car_resnet34":                     [0.5 * 255, 0.5 * 255, 0.5 * 255],
    # BGR variants (stored in BGR order)
    "resnet50_8631":                    [91.4953, 103.8827, 131.0912],
    "resnet50_8631_adv":                [91.4953, 103.8827, 131.0912],
    "resnet50_8631_vib":                [91.4953, 103.8827, 131.0912],
    "resnet50_100":                     [91.4953, 103.8827, 131.0912],
    "resnet50_100_adv":                 [91.4953, 103.8827, 131.0912],
    "resnet50_100_vib":                 [91.4953, 103.8827, 131.0912],
}
 
ALL_STDS = {
    "vgg16":                            [1.0, 1.0, 1.0],
    "vgg16_5class":                     [1.0, 1.0, 1.0],
    "vgg16_8class":                     [1.0, 1.0, 1.0],
    "vgg16_9class":                     [1.0, 1.0, 1.0],
    "vgg16_16class":                    [1.0, 1.0, 1.0],
    "vgg16_24class":                    [1.0, 1.0, 1.0],
    "vgg16_10class_dp_sgd":             [1.0, 1.0, 1.0],
    "vgg16_10class":                    [1.0, 1.0, 1.0],
    "vgg16bn":                          [1.0, 1.0, 1.0],
    "resnet50":                         [1.0, 1.0, 1.0],
    "mobilenet_v2":                     [1.0, 1.0, 1.0],
    "efficientnet_b0":                  [1.0, 1.0, 1.0],
    "inception_v3":                     [1.0, 1.0, 1.0],
    "swin_transformer":                 [1.0, 1.0, 1.0],
    "vision_transformer":               [1.0, 1.0, 1.0],
    "inception_resnetv1_vggface2":      [128.0, 128.0, 128.0],
    "inception_resnetv1_vggface2_8631": [128.0, 128.0, 128.0],
    "inception_resnetv1_casia":         [128.0, 128.0, 128.0],
    "sphere20a":                        [128.0, 128.0, 128.0],
    "efficientnet_b0_casia":            [128.0, 128.0, 128.0],
    "ccs19ami_facescrub":               [255.0, 255.0, 255.0],
    "ccs19ami_facescrub_rgb":           [255.0, 255.0, 255.0],
    "azure":                            [255.0, 255.0, 255.0],
    "cat_resnet18":                     [0.229 * 255, 0.224 * 255, 0.225 * 255],
    "resnet18_10class":                 [0.229 * 255, 0.224 * 255, 0.225 * 255],
    "car_resnet34":                     [0.5 * 255, 0.5 * 255, 0.5 * 255],
    "resnet50_8631":                    [1.0, 1.0, 1.0],
    "resnet50_8631_adv":                [1.0, 1.0, 1.0],
    "resnet50_8631_vib":                [1.0, 1.0, 1.0],
    "resnet50_100":                     [1.0, 1.0, 1.0],
    "resnet50_100_adv":                 [1.0, 1.0, 1.0],
    "resnet50_100_vib":                 [1.0, 1.0, 1.0],
}
 
# Architectures whose models expect BGR channel order rather than RGB
BGR_ARCHS = frozenset({
    "resnet50_8631",
    "resnet50_8631_adv",
    "resnet50_8631_vib",
    "resnet50_100",
    "resnet50_100_adv",
    "resnet50_100_vib",
})


####################
# Image Processing #
####################

def normalize(image_tensor, arch_name):
    """
    input image is in [0., 255.] and RGB channel
    """
    if 'resnet50_100' in arch_name or 'resnet50_8631' in arch_name:
        # change RGB to BGR
        if image_tensor.ndim == 4:
            assert image_tensor.shape[1] == 3
            image_tensor = image_tensor[:, [2, 1, 0]]
        else:
            assert image_tensor.ndim == 3
            assert image_tensor.shape[0] == 3
            image_tensor = image_tensor[[2, 1, 0]]
    std = ALL_STDS[arch_name]
    mean = ALL_MEANS[arch_name]
    image_tensor = (image_tensor-torch.tensor(mean, device=image_tensor.device)[:, None, None])/torch.tensor(std, device=image_tensor.device)[:, None, None]
    return image_tensor
 

def denormalize(image_tensor, arch_name):
    """
    output image is in [0., 1.] and RGB channel
    """
    std = ALL_STDS[arch_name]
    mean = ALL_MEANS[arch_name]
    image_tensor = image_tensor * torch.tensor(std, device=image_tensor.device)[:, None, None] + torch.tensor(mean, device=image_tensor.device)[:, None, None]
    image_tensor = image_tensor / 255.

    if 'resnet50_100' in arch_name or 'resnet50_8631' in arch_name:
        # change BGR to RGB
        if image_tensor.ndim == 4:
            assert image_tensor.shape[1] == 3
            image_tensor = image_tensor[:, [2, 1, 0]]
        else:
            assert image_tensor.ndim == 3
            assert image_tensor.shape[0] == 3
            image_tensor = image_tensor[[2, 1, 0]]

    return torch.clamp(image_tensor, 0., 1.)


def crop_img_for_sphereface(img):
    assert len(img.shape) == 3 or len(img.shape) == 4
    # resize the img to 256 first because the following crop area are defined in 256 scale
    img = F.resize(img, (256, 256))
    return img[..., 16:226, 38:218]


def crop_img(img, arch_name):
    if arch_name == 'sphere20a':
        return crop_img_for_sphereface(img)
    elif arch_name.startswith('ccs19ami'):
        raise AssertionError('do not use white-box attack for ccs19ami')
    else:
        return img
    

def resize_img(img, image_resolution):
    if not isinstance(image_resolution, tuple):
        image_resolution = (image_resolution, image_resolution)
    return F.resize(img, image_resolution)

    
def crop_and_resize(img, arch_name, resolution):
    img = crop_img(img, arch_name)
    img = resize_img(img, resolution)
    return img