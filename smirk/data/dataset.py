# smirk/data/dataset.py

"""
QueryDataset: the dataset used to train the surrogate model.
 
Each item is an image, soft label pair where the soft label is the logit vector produced by querying the target blackbox model.
The dataset expects:

    - A list of image batch tensors produced by smirk/data/sampler.py.
    - A single merged logits tensor of shape (N, num_classes) produced by scripts/generate_blackbox_attack_data.py.
"""

from typing import List, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import json

class QueryDataset(Dataset):
    """
    Dataset of image, label pairs for surrogate training
    """
    def __init__(self, image_batches: List[Tensor], logits: Tensor, transform: transforms.Compose):
        self.image_batches = image_batches
        self.batch_size = image_batches[0].shape[0] # derive batch size from first batch instead of hardcoded value
        self.transform = transform

        self.logits = logits # (N, num_classes)

    def __len__(self):
        return self.logits.shape[0]

    def __getitem__(self, idx):
        batch_index = idx // self.batch_size
        index_within_batch = idx % self.batch_size

        # load image
        image = self.image_batches[batch_index][index_within_batch]
        image = self.transform(image)


        soft_label = self.logits[idx]
        return image, soft_label
    
class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image_tensor):
        image_tensor = (image_tensor-torch.tensor(self.mean, device=image_tensor.device)[:, None, None])/torch.tensor(self.std, device=image_tensor.device)[:, None, None]
        return image_tensor

def make_train_transform(
        resolution: int,
        mean: List[float],
        std: List[float]
    ):
    """
    Returns standard training transform for QueryDataset:
        - scales image from [0,1] to [0,255]
        - resizes to specified resolution
        - normalises using specified model dependant mean and std values
    """

    def scale255(x: Tensor) -> Tensor:
        return x * 255.0
        
    return transforms.Compose([
        scale255,
        transforms.Resize(resolution),
        Normalize(mean, std)
    ])

def make_test_transform(
        resize_resolution: int,
        mean: List[float],
        std: List[float],
        crop_resolution: int = None
    ):
    """
    Returns standard test transform for evaluation:
        - scales image from [0,1] to [0,255]
        - converts PIL image to tensor
        - resizes to specified resolution
        - extracts center crop of image if necessary (depends on target dataset)
        - normalises using specified model dependant mean and std values
    """

    def scale255(x: Tensor) -> Tensor:
        return x * 255.0

    transformlist = [
        scale255,
        transforms.PILToTensor(),
        transforms.Resize(resize_resolution)
    ]

    if crop_resolution:
        transformlist.append(
            transforms.CenterCrop(crop_resolution)
        )

    transformlist.append(
        Normalize(mean, std)
    )

    return transforms.Compose(transformlist)