import torch
from omegaconf import DictConfig
from smirk.genforce import my_get_GD

# genforce GAN generator
def get_generator(sampling_config: DictConfig, device: torch.device):
    generator, _ = my_get_GD.main(
        device,
        sampling_config.model.genforce_model,
        sampling_config.batch_size,
        sampling_config.batch_size,
        use_w_space=sampling_config.latent_space.use_w_space,
        use_discri=False,
        repeat_w=sampling_config.latent_space.repeat_w,
        use_z_plus_space=sampling_config.latent_space.use_z_plus_space,
        trunc_psi=sampling_config.latent_space.trunc_psi,
        trunc_layers=sampling_config.latent_space.trunc_layers,
    )
    return generator