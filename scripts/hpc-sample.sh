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

#SBATCH --output=/sharedscratch/SMIRK/jobs/job_output_%j.log   # Standard output (print statements)
#SBATCH --error=/sharedscratch/SMIRK/jobs/job_error_%j.log     # Errors/Tracebacks

# activate environment
module load conda
conda activate SMIRK-HPC
cd sharedscratch/SMIRK

# run py script
source activate pytorch
python my_sample_z_w_space.py &
PID=$!

# output nvidia interface
while ps -p $PID > /dev/null;
do
nvidia-smi > smi_0.txt
sleep 10
done