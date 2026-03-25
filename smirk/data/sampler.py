#!/usr/bin/env python3
import os
import glob
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from torchvision.utils import save_image

def get_generator(cfg: DictConfig, batch_size: int, device: torch.device):
    from genforce import my_get_GD
    generator, _ = my_get_GD.main(
        device,
        cfg.model.genforce_model,
        batch_size,
        batch_size,
        use_w_space=cfg.latent.use_w_space,
        use_discri=False,
        repeat_w=cfg.latent.repeat_w,
        use_z_plus_space=cfg.latent.use_z_plus_space,
        trunc_psi=cfg.latent.trunc_psi,
        trunc_layers=cfg.latent.trunc_layers,
    )

# Ensure codependant flags for use w_space and repeat_w are consistent
# use_z_plus_space requires use_w_space=true, repeat_w=false
def validate_latent_config(cfg: DictConfig):
    if cfg.latent.use_z_plus_space:
        assert cfg.latent.use_w_space
        assert not cfg.latent.repeat_w
