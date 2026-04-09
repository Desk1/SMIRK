# smirk/data/sampler.py - generate GAN samples

"""
Sampler: Samples random latent vectors from a specified GAN model and saves the corresponding image.
"""

import os
import glob
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from pathlib import Path
from smirk.genforce import my_get_GD

class Sampler():
    def __init__(
        self,
        device: torch.device,
        generator: my_get_GD.Fake_G,
        config: DictConfig
    ):
        self.device = device
        self.config = config
        self.generator = generator

        self.manifest_data = {
            "model": self.config.model.genforce_model,
            "truncation_psi": self.config.latent_space.trunc_psi,
            "truncation_layers": self.config.latent_space.trunc_layers,
            "requested_iterations": self.config.size,
            "batch_size": self.config.batch_size,
            "latent_dim": self.config.latent_dim,
            "device": str(device),
            "completed_iterations": 0,
            "status": "",
            "batch_files": []
        }

    @torch.no_grad()
    def generate_samples(self, output_dir: Path):

        for i in tqdm(range(1, self.config.size + 1), desc="Sampling"):
            # generate latent vector
            latent_in = torch.randn(self.config.batch_size, self.config.latent_dim, device=self.device)
            filename = output_dir / f"sample_{i}"

            # save vector + image
            img_gen = self.generator(latent_in)
            torch.save(img_gen, f"{filename}_img.pt")
            torch.save(latent_in, f"{filename}_latent.pt")

            # update manifest 
            self.manifest_data["completed_iterations"] += 1
            self.manifest_data["batch_files"].append({
                "iteration": i,
                "image_file": f"sample_{i}_img.pt",
                "latent_file": f"sample_{i}_latent.pt"
            })

    
    def merge_vectors(self, output_dir: Path):
        # Collect all_ws.pt
        
        all_ws = []
        latent_files = sorted(glob.glob(f"{output_dir}/sample_*_latent.pt"))
        for i in tqdm(range(0, len(latent_files), self.config.batch_size), desc="Merging"):
            batch_files = latent_files[i:i+self.config.batch_size]
            print(batch_files)
            latent_in = [torch.load(f) for f in batch_files]
            latent_in = torch.cat(latent_in, dim=0)
            w = self.generator.G.mapping(latent_in.to(self.device))["w"]
            all_ws.append(w)

        all_ws = torch.cat(all_ws, dim=0).cpu()
        torch.save(all_ws, f"{output_dir}/all_ws.pt")

        self.manifest_data["status"] = "completed" if self.manifest_data["completed_iterations"] == self.config.size else "interrupted"
