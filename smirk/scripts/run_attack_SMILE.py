import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
import json
import logging
from omegaconf import DictConfig
from typing import Dict

import smirk.models
from smirk.models.registry import get_model, get_expert_model, get_spec
from smirk.utils.files import *
from smirk.genforce import my_get_GD
from smirk.attacks.base import AttackResult
from smirk.attacks.whitebox import SMILEWhiteboxAttack
from smirk.attacks.blackbox import SMILEBlackboxAttack
from smirk.attacks.population import VectorizedPopulation

log = logging.getLogger(__name__)

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

def get_test_model_name(target_model_name: str):
    """
    Map of suitable test models for given target models
    """
    match target_model_name:
        case 'vgg16bn':
            return 'vgg16'
        case 'vgg16':
            return 'vgg16bn'
        case 'resnet50':
            return 'inception_resnetv1_vggface2'
        case 'inception_resnetv1_vggface2':
            return 'resnet50'
        case 'mobilenet_v2':
            return 'resnet50'
        case 'efficientnet_b0':
            return 'resnet50'
        case 'inception_v3':
            return 'resnet50'
        case 'swin_transformer':
            return 'resnet50'
        case 'vision_transformer':
            return 'resnet50'
        case 'inception_resnetv1_casia':
            return 'efficientnet_b0_casia'
        case 'efficientnet_b0_casia':
            return 'inception_resnetv1_casia'
        case 'sphere20a':
            return 'inception_resnetv1_casia'
        case _:
            log.warning(
                f"No test model assigned for {target_model_name}, using default (resnet50)"
            )
            return 'resnet50'
        
        
def run_whitebox_attack(
    all_ws_file,
    all_logits_file,
    surrogate_model,
    surrogate_model_spec,
    test_model,
    test_model_spec,
    generator,
    writer,
    device,
    config
):
    """
    Build population and execute whitebox attack for each target
    """
    results: Dict[int, AttackResult] = {} # target_label : result

    for target_label in config.attack_execution.targets:
        log.info(f"attacking target: {target_label}")

        population = VectorizedPopulation(
            all_ws = torch.load(all_ws_file, map_location=device),
            all_logits = torch.load(all_logits_file, map_location=device),
            population_size = config.attack_execution.whitebox_attack.population_size,
            target_label = target_label
        )
    
        SMILE = SMILEWhiteboxAttack(
            target_model = surrogate_model,
            target_model_spec = surrogate_model_spec,
            test_model = test_model,
            test_model_spec = test_model_spec,
            generator = generator,
            writer = writer,
            device = device,
            population = population,
            epochs = config.attack_execution.whitebox_attack.epochs,
            learning_rate = config.attack_execution.whitebox_attack.learning_rate
        )

        try:
            result = SMILE.run(target_label)
        except Exception as e:
            log.error(
                f"Error during whitebox attack against target {target_label}:\n{e}"
            )
            result = None

        results[target_label] = result

    return results

def run_blackbox_attack(
    all_ws_file,
    all_logits_file,
    target_model,
    target_model_spec,
    test_model,
    test_model_spec,
    generator,
    writer,
    device,
    config,
    whitebox_attack_results
):
    """
    Build population and execute whitebox attack for each target
    """
    results: Dict[int, AttackResult] = {} # target_label : result

    for target_label in config.attack_execution.targets:
        log.info(f"attacking target: {target_label}")

        # load elite starting point from whitebox attack
        elite = whitebox_attack_results[target_label].latent_vector

        population = VectorizedPopulation(
            all_ws = torch.load(all_ws_file, map_location=device),
            all_logits = torch.load(all_logits_file, map_location=device),
            population_size = config.attack_execution.blackbox_attack.population_size,
            target_label = target_label
        )
    
        SMILE = SMILEBlackboxAttack(
            target_model = target_model,
            target_model_spec = target_model_spec,
            test_model = test_model,
            test_model_spec = test_model_spec,
            generator = generator,
            writer = writer,
            device = device,
            population = population,
            epochs = config.attack_execution.blackbox_attack.epochs,
            learning_rate = config.attack_execution.blackbox_attack.learning_rate,
            elite_vector = elite,
            budget = config.attack_execution.blackbox_attack.budget,
            optimizer_strategy = config.attack_execution.blackbox_attack.optimizer_strategy
        )

        result = SMILE.run(target_label)

        results[target_label] = result

    return results


@hydra.main(config_path=str(get_path("configs")), config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # build surrogate model + specification
    surrogate_dir = get_surrogate_training_directory(config)
    manifest_path = surrogate_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        
        surrogate_model = get_expert_model(
            name = manifest["model_architecture"],
            num_experts = manifest["num_experts"],
            num_classification= manifest["num_classification"],
            device = device,
            load_weights=False
        )
        surrogate_model.load_state_dict(
            torch.load(str(surrogate_dir / "best_model.pt"), map_location=device)
        )
        surrogate_model_spec = get_spec(manifest["model_architecture"])
    else:
        raise FileNotFoundError(
            f"Could not find surrogate manifest file at {manifest_path}. "
            "Ensure surrogate training script has been ran for this config"
        )
    
    # build test model + specification
    test_model_name = get_test_model_name(surrogate_model_spec.name)
    test_model = get_model(test_model_name, device)
    test_model_spec = get_spec(test_model_name)

    # build target model + specification
    target_model = get_model(config.blackbox_sample_query.arch_name_target, device)
    target_model_spec = get_spec(config.blackbox_sample_query.arch_name_target)

    # todo: setup forward hooks for test model

    # GAN generator
    generator = get_generator(config.sampling, device)

    # output
    output_dir = get_attack_execution_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))

    # load sampling + blackbox attack data
    sample_dir = get_sampling_directory(config)
    blackbox_attack_data_dir = get_blackbox_attack_data_directory(config)
    all_ws_file = sample_dir / "all_ws.pt"
    all_logits_file = blackbox_attack_data_dir / "all_logits.pt"

    if not all_ws_file.exists():
        raise FileNotFoundError(
            f"Could not find sampling data at {all_ws_file}. "
            "Ensure sampling script has been ran for this config"
        )
    if not all_logits_file.exists():
        raise FileNotFoundError(
            f"Could not find blackbox attack data at {all_logits_file}. "
            "Ensure generate blackbox attack data script has been ran for this config"
        )
    
    whitebox_results = run_whitebox_attack(
        all_ws_file,
        all_logits_file,
        surrogate_model,
        surrogate_model_spec,
        test_model,
        test_model_spec,
        generator,
        writer,
        device,
        config
    )

    blackbox_results = run_blackbox_attack(
        all_ws_file,
        all_logits_file,
        target_model,
        target_model_spec,
        test_model,
        test_model_spec,
        generator,
        writer,
        device,
        config,
        whitebox_attack_results=whitebox_results
    )

    for target in blackbox_results:
        log.info(f"target {target}:")
        log.info(blackbox_results[target])

        save_dir = output_dir / f"{target}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for r in blackbox_results[target]:
            torch.save(blackbox_results[target][r], f"{r}.pt")

    
if __name__ == "__main__":
    main()