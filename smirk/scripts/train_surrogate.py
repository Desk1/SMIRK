from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import smirk.models
from smirk.data.dataset import QueryDataset, make_train_transform, make_test_transform
from smirk.models.registry import get_expert_model, get_spec
from smirk.training.long_tail import build_weight_k
from smirk.training.trainer import SurrogateTrainer
from smirk.utils.files import *

log = logging.getLogger(__name__)


@hydra.main(config_path=str(get_path("configs")), config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    set_seed(config.seed)

    # load logits
    logits_path = get_blackbox_attack_data_directory(config) / "all_logits.pt"
    if logits_path.exists():
        all_logits = torch.load(logits_path, map_location=device)
        all_logits = all_logits[: config.surrogate_training.query_num]
    else:
        raise FileNotFoundError(
            f"{logits_path} could not be found. "
            "run smirk/scripts/generate_blackbox_attack_data.py for this config first"
        )
    
    # build surrogate model + specification
    model = get_expert_model(
        name = config.surrogate_training.arch_name_surrogate,
        num_experts = config.surrogate_training.num_experts,
        num_classification= all_logits.shape[1],
        device = device
    )
    spec = get_spec(config.surrogate_training.arch_name_surrogate)

    # build optimizer
    paramlist = [{'params': x} for x in model.modified_layer_parameters]
    optimizer = torch.optim.Adam(paramlist, lr=0.001)
    
    # load img files
    sample_dir = get_sampling_directory(config)
    img_files = get_sample_images(sample_dir)
    img_files = img_files[: int(config.surrogate_training.query_num / 100.0)]
    img_files = [torch.load(img).to(device) for img in img_files]

    # build training data
    train_transform = make_train_transform(spec.resolution, spec.mean, spec.std)
    train_dataset = QueryDataset(img_files, all_logits, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.surrogate_training.batch_size,
        shuffle=True,
    )

    # build test data - target dataset specific
    dataset = None
    match config.blackbox_sample_query.target_dataset:
        case "CASIA":
            test_transform = make_test_transform(
                resize_resolution = spec.resolution,
                mean = spec.mean,
                std = spec.std
            )
            dataset = "CASIA-WebFace"
        
        case "vggface":
            test_transform = make_test_transform(
                resize_resolution = spec.resolution+1,
                crop_resolution = spec.resolution,
                mean = spec.mean,
                std = spec.std
            )
        
        case "vggface2":
            test_transform = make_test_transform(
                resize_resolution = 300,
                crop_resolution = spec.resolution,
                mean = spec.mean,
                std = spec.std
            )
            dataset = "vggface2/train"
        
        case _: test_transform = None
    
    test_loader = None
    if config.surrogate_training.test_surrogate:
        dataset_dir = get_dataset_directory(dataset)
        if dataset_dir.exists():
            totalset = torchvision.datasets.ImageFolder(dataset_dir, transform=test_transform)
            _trainset_list, testset_list = train_test_split(list(range(len(totalset.samples))), test_size=0.01, random_state=666)
            testset= Subset(totalset, testset_list)
            test_loader = DataLoader(
                testset,
                batch_size=config.surrogate_training.batch_size,
                shuffle=False,
                num_workers=8, 
                pin_memory=True
            )
            
        else:
            raise FileNotFoundError(
                f"Dataset not found at {dataset_dir}"
            )


    # build long-tailed weighting values
    topk_weights = build_weight_k(all_logits, beta=config.surrogate_training.beta)

    # output
    output_dir = get_surrogate_training_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))

    # trainer
    trainer = CheckpointingSurrogateTrainer(
        model = model,
        device = device,
        train_loader = train_loader,
        test_loader = test_loader,
        output_dir = output_dir,
        writer = writer,
        topk_weights = topk_weights,
        epochs = config.surrogate_training.epochs,
        num_experts = config.surrogate_training.num_experts,
        lambda_diversity = config.surrogate_training.lambda_diversity,
        lambda_ce = config.surrogate_training.lambda_ce
    )

    try:
        model = trainer.fit()
    finally:
        writer.close()
        torch.save(model.state_dict(), output_dir / "final_model.pt")
        
        # save manifest with surrogate model metadata
        manifest = {
            "model_architecture": config.surrogate_training.arch_name_surrogate,
            "num_experts": config.surrogate_training.num_experts,
            "num_classification": all_logits.shape[1]
        }
        
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        log.info(f"Training complete. Checkpoints and manifest saved to {output_dir}")


class CheckpointingSurrogateTrainer(SurrogateTrainer):
    """Extends SurrogateTrainer with checkpoint saving via the hook methods"""

    def __init__(self, *args, output_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir

    def on_best(self, epoch: int, acc: float):
        path = self.output_dir / "best_model.pt"
        torch.save(self.model.state_dict(), path)

    def on_epoch_end(self, epoch: int):
        pass


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()