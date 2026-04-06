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
from smirk.models.stats import ALL_MEANS, ALL_STDS


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