#!/bin/bash --login
#SBATCH --job-name=sample-synthetic-images
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=12:00:00

#SBATCH --output=jobs/job_b_output_%j.log
#SBATCH --error=jobs/job_b_error_%j.log

# activate environment
conda activate SMIRK-HPC

# run py script
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_sample_z_w_space.py 
