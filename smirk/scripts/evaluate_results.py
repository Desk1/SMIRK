import torch
import logging
import hydra
import csv
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, List

import smirk.models
from smirk.models.registry import get_model, get_spec
from smirk.utils.files import get_path, get_attack_execution_directory, get_surrogate_training_directory
from smirk.utils.models import get_test_model_name
from smirk.evaluation.metrics import ASR
from smirk.scripts.generate_report import generate_report, build_label_to_folder

log = logging.getLogger(__name__)


def compute_asr(generated_image: torch.Tensor, test_model, target_label: int, device: torch.device) -> bool:
    """
    Compute Attack Success Rate by passing generated image through test model
    
    Args:
        generated_image: Generated adversarial image
        test_model: Test/victim model
        target_label: Target class label
        device: Device to use for computation
        
    Returns:
        Boolean indicating if attack was successful
    """
    with torch.no_grad():
        # Ensure image is on correct device
        img = generated_image.to(device)
        
        # Add batch dimension if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # Get predictions from test model
        predictions = test_model(img)
        
        # Compute ASR
        asr_result = ASR(predictions, target_label)
    
    return asr_result


@hydra.main(config_path=str(get_path("configs")), config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    attack_dir = get_attack_execution_directory(config)
    blackbox_dir = attack_dir / "blackbox"

    # build test model + specification
    test_model_name = get_test_model_name(config.surrogate_training.arch_name_surrogate)
    test_model = get_model(test_model_name, device)
    test_model_spec = get_spec(test_model_name)
    test_model.eval()

    # Collect results
    evaluation_results: List[Dict] = []

    # Iterate through target directories
    target_dirs = sorted([d for d in blackbox_dir.iterdir() if d.is_dir() and d.name.startswith("target_")])
    
    if not target_dirs:
        log.warning(f"No target directories found in {blackbox_dir}")
        return

    for target_dir in target_dirs:
        target_label = int(target_dir.name.split("_")[1])
        log.info(f"Processing target {target_label}")

        # Load all result files in this target directory
        result_files = sorted(target_dir.glob("*.pt"))
        
        if not result_files:
            log.warning(f"No result files found in {target_dir}")
            continue

        # Assuming there's one result file per target (e.g., 'result.pt' or similar)
        # or multiple result files from different attack runs
        for result_file in result_files:
            try:
                result_data = torch.load(result_file, map_location=device)
                
                generated_image = result_data.generated_image
                latent_vector = result_data.latent_vector
                fitness_score = result_data.fitness_score

                if generated_image is None:
                    log.warning(f"No generated image found in {result_file}")
                    continue

                # Compute ASR
                asr = compute_asr(generated_image, test_model, target_label, device)

                # Store result
                evaluation_results.append({
                    "target_label": target_label,
                    "result_file": result_file.name,
                    "asr": int(asr),  # Convert bool to int for CSV
                    "fitness_score": fitness_score if fitness_score is not None else "N/A"
                })

                log.info(f"  {result_file.name}: ASR={asr}, fitness_score={fitness_score}")

            except Exception as e:
                log.error(f"Error processing {result_file}: {e}")
                continue

    # Write results to CSV
    if evaluation_results:
        csv_path = attack_dir / "evaluation_results.csv"
        
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["target_label", "result_file", "asr", "fitness_score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(evaluation_results)
        
        log.info(f"Evaluation results saved to {csv_path}")

        # Print summary statistics
        total_results = len(evaluation_results)
        successful_attacks = sum(1 for r in evaluation_results if r["asr"] == 1)
        asr_rate = successful_attacks / total_results if total_results > 0 else 0

        log.info(f"Summary: {successful_attacks}/{total_results} successful attacks (ASR: {asr_rate:.2%})")

        # Generate HTML report
        output_dir = get_path("output")
        target_dataset = config.blackbox_sample_query.target_dataset
        dataset_train_dir = get_path("datasets") / target_dataset / "train"
        label_to_folder = build_label_to_folder(dataset_train_dir) if dataset_train_dir.exists() else None
        report_path = generate_report(output_dir, dataset_train_dir if dataset_train_dir.exists() else None, label_to_folder)
        log.info(f"HTML report: file://{report_path}")
    else:
        log.warning("No evaluation results computed")


if __name__ == "__main__":
    main()