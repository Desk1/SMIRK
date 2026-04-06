# smirk/data/sampler.py - generate GAN samples

import os
import glob
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from torchvision.utils import save_image
from smirk.utils.files import get_path

def get_generator(cfg: DictConfig, batch_size: int, device: torch.device):
    from smirk.genforce import my_get_GD
    generator, _ = my_get_GD.main(
        device,
        cfg.model.genforce_model,
        batch_size,
        batch_size,
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

@torch.no_grad()
@hydra.main(config_path=get_path("configs"), config_name="sampling", version_base=None)
def sample(cfg: DictConfig):
    validate_latent_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = cfg.batch_size
    generator = get_generator(cfg, batch_size, device)

    iter_times = cfg.size
    dirname = (
        f"{cfg.output_dir}/"
        f"{cfg.model.genforce_model}_"
        f"{cfg.latent_space.trunc_psi}_"
        f"{cfg.latent_space.trunc_layers}_"
        f"{cfg.size}"
    )

    os.makedirs(dirname, exist_ok=True)

    signal_file = f"{dirname}/signal"

    for i in tqdm(range(1, iter_times + 1)):
        if not os.path.isfile(signal_file):
            with open(signal_file, "w") as f:
                f.write("0")
        with open(signal_file) as f:
            if f.readline().strip() == "1":
                print("Stop iteration now")
                break

        latent_in = torch.randn(batch_size, cfg.latent_dim, device=device)
        filename = f"{dirname}/sample_{i}"

        img_gen = generator(latent_in)
        torch.save(img_gen, f"{filename}_img.pt")
        torch.save(latent_in, f"{filename}_latent.pt")

    # Collect all_ws.pt
    all_ws = []
    for i in tqdm(range(0, len(sorted(glob.glob(f"{dirname}/sample_*_latent.pt"))), batch_size)):
        latent_files = sorted(glob.glob(f"{dirname}/sample_*_latent.pt"))[i:i+batch_size]
        latent_in = [torch.load(f) for f in latent_files]
        latent_in = torch.cat(latent_in, dim=0)
        w = generator.G.mapping(latent_in.to(device))["w"]
        all_ws.append(w)

    all_ws = torch.cat(all_ws, dim=0).cpu()
    torch.save(all_ws, f"{dirname}/all_ws.pt")


if __name__ == "__main__":
    sample()