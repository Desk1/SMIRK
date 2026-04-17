# smirk/utils/files.py - file handling and logging utils

# API
# ---------------------------------
# get_path(relative)               # get path to relative location from project root
# create_folder(folder)            # create folder(s)

import os, sys, json
from pathlib import Path
from omegaconf import DictConfig
from typing import List


def get_path(relative_location: str) -> Path:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    return PROJECT_ROOT / relative_location

def get_sampling_directory(cfg: DictConfig) -> Path:
    dirname = (
        f"{cfg.sampling.output_dir}/"
        f"{cfg.sampling.model.genforce_model}_"
        f"{cfg.sampling.latent_space.trunc_psi}_"
        f"{cfg.sampling.latent_space.trunc_layers}_"
        f"{cfg.sampling.size}"
    )

    return get_path(dirname)


def get_blackbox_attack_data_directory(cfg: DictConfig) -> Path:
    sample_dataset = (
        f"{cfg.sampling.model.genforce_model}_"
        f"{cfg.sampling.latent_space.trunc_psi}_"
        f"{cfg.sampling.latent_space.trunc_layers}_"
        f"{cfg.sampling.size}"
    )

    dirname = (
        f"{cfg.blackbox_sample_query.output_dir}/"
        f"{cfg.blackbox_sample_query.target_dataset}/"
        f"{cfg.blackbox_sample_query.arch_name_target}/"
        f"{sample_dataset}/"
    )

    return get_path(dirname)

def get_surrogate_training_directory(cfg: DictConfig) -> Path:
    dirname = (
        f"{cfg.surrogate_training.output_dir}/"
        f"{cfg.surrogate_training.arch_name_surrogate}"
    )

    return get_path(dirname)

def get_attack_execution_directory(cfg: DictConfig) -> Path:
    dirname = (
        f"{cfg.attack_execution.output_dir}/"
        f"{cfg.attack_execution.attack_mode}/"
        f"{cfg.surrogate_training.arch_name_surrogate} -> {cfg.blackbox_sample_query.arch_name_target}"
    )

    return get_path(dirname)

def get_dataset_directory(dataset: str) -> Path:
    dirname = (
        "datasets/"
        f"{dataset}"
    )

    return get_path(dirname)

def get_sample_images(sample_dir: Path) -> List[str]:
    img_files = []
    manifest_path = sample_dir / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        
        img_files = sorted(sample_dir / batch["image_file"] for batch in manifest["batch_files"])
    else:
        raise FileNotFoundError(
            f"No sample manifest found at {manifest_path}. "
            "Run smirk/scripts/sample.py to generate a for this config first"
        )
    
    return img_files
