# smirk/scripts/generate_blackbox_attack_data.py

"""
Builds the query dataset used to train the surrogate model.
 
This script replaces two separate scripts from the SMILE codebase:
 
    - my_generate_blackbox_attack_dataset.py  (inference pass)
    - my_merge_all_tensors.py                 (merge pass)
 
Pipeline
--------
1. Inference pass: for each image batch saved in samples/, run it through
the target model and save a corresponding *_logits.pt file in the output directory.
 
2. Merge pass: concatenate all batch logit files into a single all_logits.pt file
 
Output layout
-------------
blackbox_attack_data/
    <target_dataset>/
        <arch_name>/
            <sampling_dataset>/
                sample_1_img_logits.pt   # intermediate batch logits
                sample_2_img_logits.pt
                ...
                all_logits.pt            # final merged logits
 
Usage
-----
# Configure all parameters from configs/...

python -m smirk.scripts.2-generate_blackbox_attack_data.py
"""

import torch
import hydra
import json
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

import smirk.models
from smirk.models.registry import get_model, get_resolution
from smirk.utils.files import get_path, get_sampling_directory, get_blackbox_attack_data_directory, get_sample_images
from smirk.utils.image import normalize, resize_img, crop_img

log = logging.getLogger(__name__)

@torch.no_grad()
def run_inference(
    arch_name: str,
    resolution: Union[int, tuple],
    sample_dir: Path,
    output_dir: Path,
    device: torch.device
):
    """
    Run target model inference on every image batch and save logits:
        - arch_name: name of target model architecture
        - resolution: img resolution for target model
        - sample_dir: path to sample directory
        - output_dir: path to output directory
    """
    img_files = get_sample_images(sample_dir)
    
    model = get_model(arch_name, device)
    model = model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(img_files, desc="Inference"):
        save_path = output_dir / (img_file.stem + "_logits.pt")

        img = torch.load(img_file).to(device)
        img = crop_img(img, arch_name)
        img = normalize(resize_img(img*255., resolution), arch_name)

        logits = model(img)
        if arch_name == "sphere20a":
            logits = logits[0]

        torch.save(logits, save_path)


def run_merge(output_dir: Path, remove: bool):
    """
    Concatenate intermediate batch logit files into a single all_logits.pt

    config:
        - remove: bool indicating whether to remove intermediate files
    """
    logit_files = sorted(output_dir.glob("sample_*_img_logits.pt"))

    if not logit_files:
        raise RuntimeError(
            f"No intermediate logit files found in {output_dir}"
            f"Run the iference pass first"
        )
    
    all_logits = []
    for f in tqdm(logit_files, desc="Merge"):
        all_logits.append(torch.load(f, map_location="cpu"))
    
    all_logits = torch.cat(all_logits, dim=0)

    torch.save(all_logits, output_dir / "all_logits.pt")

    if remove:
        for f in logit_files:
            f.unlink()

@hydra.main(config_path=str(get_path("configs")), config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    arch_name = config.blackbox_sample_query.arch_name_target
    remove_intermediate = config.blackbox_sample_query.remove_intermediate

    sample_dir = get_sampling_directory(config)
    output_dir = get_blackbox_attack_data_directory(config)
    resolution = get_resolution(arch_name)

    # merge only
    if config.blackbox_sample_query.merge_only:
        run_merge(output_dir, remove_intermediate)
        return

    run_inference(arch_name, resolution, sample_dir, output_dir, device)

    # inference + merge
    if not config.blackbox_sample_query.inference_only:
        run_merge(output_dir, remove_intermediate)

    log.info(
        f"Finished querying {arch_name} to {output_dir}"
    )

if __name__ == "__main__":
    main()