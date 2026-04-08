# smirk/scripts/generate_blackbox_attack_data.py

"""
Builds the query dataset used to train the surrogate model.
 
This script replaces two separate scripts from the SMILE codebase:
 
    - my_generate_blackbox_attack_dataset.py  (inference pass)
    - my_merge_all_tensors.py                 (merge pass)
 
The two passes are run sequentially by default.
Pass --inference-only or --merge-only arguments to run a single step
 
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

# todo redo config info
# Full run (inference + merge):
python scripts/build_query_dataset.py

# Inference only:
python scripts/build_query_dataset.py --inference-only

# Merge only (after inference is complete):
python scripts/build_query_dataset.py --merge-only
"""

import argparse
import torch
import hydra
import json
import os
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

import smirk.models
from smirk.models.registry import get_model, get_resolution
from smirk.utils.files import get_path, get_sampling_directory, get_blackbox_attack_data_directory
from smirk.utils.image import normalize, resize_img, crop_img

@torch.no_grad()
def run_inference(
    config: DictConfig,
    device: torch.device
):
    """
    Run target model inference on every image batch and save logits.

    config:
        - arch_name: name of target model architecture
        - sampling_dataset: name of StyleGAN sample dataset
        - output_dir: (optional) manually set output directory
    """
    sample_dir = get_sampling_directory(config)
    output_dir = get_blackbox_attack_data_directory(config)
    resolution = get_resolution(config.arch_name)
    manifest_path = sample_dir / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        
        img_files = sorted(sample_dir / batch for batch in manifest["batch_files"])
    else:
        raise FileNotFoundError(
            f"Manifest file not found within {sample_dir}"
            "Run smirk/scripts/sample.py to generate a manifest"
        )
    
    model = get_model(config.arch_name, device)

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(img_files, desc="Inference"):
        save_path = output_dir / (img_file.stem + "_logits.pt")

        img = torch.load(img_file).to(device)
        img = crop_img(img, config.arch_name)
        img = normalize(resize_img(img*255., resolution), config.arch_name)

        logits = model(img)
        if config.arch_name == "sphere20a":
            logits = logits[0]

        torch.save(logits, save_path)


def run_merge(config: DictConfig):
    """
    Concatenate intermediate batch logit files into a single all_logits.pt

    config:
        - remove: bool indicating whether to remove intermediate files
    """
    output_dir = get_blackbox_attack_data_directory(config)
    logit_files = sorted(output_dir.glob("sample_*_img_logits.pt"))

    if not logit_files:
        raise RuntimeError(
            f"No intermediate logit files found in {output_dir}"
            f"Run the iference pass first"
        )
    
    all_logits = []
    for f in tqdm(logit_files, desc="Merge"):
        all_logits.apppend(torch.load(f, map_location="cpu"))
    
    all_logits = torch.cat(all_logits, dim=0)

    if config.remove:
        for f in logit_files:
            f.unlink()

@hydra.main(config_path=get_path("configs"), config_name="blackbox_query")
def main(config: DictConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # merge only
    if config.merge_only:
        run_merge(config)
        return

    run_inference(config, device)

    # inference + merge
    if not config.inference_only:
        run_merge(config)

if __name__ == "__main__":
    main()