# smirk/data/dataset.py

"""
QueryDataset: the dataset used to train the surrogate model.
 
Each item is an (image, soft_label) pair where the soft label is the logit vector produced by querying the target blackbox model.
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
    Dataset of (image, soft_label) pairs for surrogate training
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

def make_transform(resolution: int, mean: List[float], std: List[float]) -> transforms.Compose:
    """
    Returns standard transform for QueryDataset:
        - scales image from [0,1] to [0,255]
        - resizes to specified resolution
        - normalises using specified model dependant mean and std values
    """

    def scale255(x: Tensor) -> Tensor:
        return x * 255.0

    class Normalize(torch.nn.Module):

        def __init__(self, mean, std):
            super().__init__()
            self.mean = mean
            self.std = std

        def forward(self, image_tensor):
            image_tensor = (image_tensor-torch.tensor(self.mean, device=image_tensor.device)[:, None, None])/torch.tensor(self.std, device=image_tensor.device)[:, None, None]
            return image_tensor
        
    return transforms.Compose([
        scale255,
        transforms.Resize(resolution),
        Normalize(mean, std)
    ])

def build_query_dataset(sample_dir: Path, logits_path: Path, transform: transforms.Compose, device: str) -> QueryDataset:
    """
    Builds a QueryDataset by loading files from disk:
        - sample_dir: directory containing image batch samples and the manifest
        - logits_path: path to all_logits.pt file containing merged logits tensor
    """

    # validate manifest
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {sample_dir}. "
            "Run smirk/data/sampler.py to regenerate the sample directory."
        )

    with manifest_path.open() as f:
        manifest = json.load(f)

    expected_files = manifest["batch_files"]
    missing = [batch for batch in expected_files if not (sample_dir / batch['image_file']).exists()]
    if missing:
        raise FileNotFoundError(
            f"Manifest lists {len(expected_files)} expected files but {len(missing)} are missing"
        )

    image_batches = [torch.load(sample_dir / batch['image_file'], map_location=device) for batch in expected_files]

    logits = torch.load(logits_path, map_location=device)    

    return QueryDataset(image_batches, logits, transform)
