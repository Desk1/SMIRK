#!/bin/bash --login
#SBATCH --job-name=sample-synthetic-images
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#SBATCH --output=jobs/job_output_%j.log   # Standard output (print statements)
#SBATCH --error=jobs/job_error_%j.log     # Errors/Tracebacks



# activate environment
module load conda
conda activate SMIRK-HPC

# run py script
python scripts/hpc-gpudebug.py