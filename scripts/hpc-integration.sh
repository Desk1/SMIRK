#!/bin/bash --login
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=1:30:00

#SBATCH --output=jobs/job_output_%j.log
#SBATCH --error=jobs/job_error_%j.log  



# activate environment
conda activate SMIRK-HPC

# run scripts
python -m smirk.scripts.sample

python -m smirk.scripts.query_target_model

python -m smirk.scripts.train_surrogate

python -m smirk.scripts.run_attack_SMILE

python -m smirk.scripts.evaluate_results