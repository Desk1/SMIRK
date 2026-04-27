#!/bin/bash --login
#SBATCH --job-name=MI_attack
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=24:00:00

#SBATCH --output=jobs/job_output_%j.log
#SBATCH --error=jobs/job_error_%j.log  


# activate environment
conda activate SMIRK-HPC

# run scripts
python -m smirk.scripts.run_attack_SMILE

python -m smirk.scripts.evaluate_results.py
