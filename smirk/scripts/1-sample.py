# smirk/scripts/sample.py

"""
Generates an initial pool of synthetic images by sampling from a StyleGAN model.

This script uses the smirk.data.sampler module to synthesize a dataset of face images by sampling latent vectors from a pre-trained StyleGAN generator.
The generated images and their corresponding latent codes are saved.

Output layout
-------------
samples/
    <model_name>_<truncation>_<trunc_layers>_<num_samples>/
        sample_1_img.pt          # synthetic image (tensor)
        sample_1_latent.pt       # corresponding latent code
        sample_2_img.pt
        sample_2_latent.pt
        ...
        all_ws.pt                # concatenated latent codes

Usage
-----
# Configure sampling parameters in configs/latent_space/ or configs/sampling.yaml

python scripts/1-sample.py
"""

import torch
import hydra
import json
from omegaconf import DictConfig
from smirk.utils.files import get_path, get_sampling_directory
from smirk.genforce import my_get_GD
from smirk.data.sampler import Sampler

def get_generator(cfg: DictConfig, device: torch.device):
    generator, _ = my_get_GD.main(
        device,
        cfg.model.genforce_model,
        cfg.batch_size,
        cfg.batch_size,
        use_w_space=cfg.latent_space.use_w_space,
        use_discri=False,
        repeat_w=cfg.latent_space.repeat_w,
        use_z_plus_space=cfg.latent_space.use_z_plus_space,
        trunc_psi=cfg.latent_space.trunc_psi,
        trunc_layers=cfg.latent_space.trunc_layers,
    )
    return generator

# use_z_plus_space requires use_w_space=true, repeat_w=false
def validate_latent_config(cfg: DictConfig):
    if cfg.latent_space.use_z_plus_space:
        assert cfg.latent_space.use_w_space
        assert not cfg.latent_space.repeat_w

@hydra.main(config_path=str(get_path("configs")), config_name="config", version_base=None)
def main(config: DictConfig):
    validate_latent_config(config.sampling)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    generator = get_generator(config.sampling, device)

    sampler = Sampler(device, generator, config.sampling)
    output_dir = get_sampling_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler.generate_samples(output_dir)
    sampler.merge_vectors(output_dir)

    manifest_data = sampler.manifest_data
    manifest_path = output_dir / "manifest.json"

    if manifest_data["status"] == "completed":
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)
            print(
                f"Finished sampling {config.sampling.model.genforce_model} "
                f"to {output_dir}"
            )
    else:
        raise RuntimeError(
            f"Sampling failed with status {manifest_data['status']}"
        )

if __name__ == "__main__":
    main()