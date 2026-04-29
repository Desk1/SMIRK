#!/bin/bash --login
#SBATCH --job-name=query-blackbox-model
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=1:00:00

#SBATCH --output=jobs/job_b_output_%j.log
#SBATCH --error=jobs/job_b_error_%j.log

# activate environment
conda activate SMIRK-HPC

# run py script
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_generate_blackbox_attack_dataset.py --arch_name inception_resnetv1_vggface2 vggface2 celeba_partial256 