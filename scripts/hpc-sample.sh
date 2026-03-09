#!/bin/bash --login
#SBATCH --job-name=sample-synthetic-images
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=1:00:00

#SBATCH --output=jobs/job_output_%j.log
#SBATCH --error=jobs/job_error_%j.log  



# activate environment
conda activate SMIRK-HPC

# run py script
python my_sample_z_w_space.py &
PID=$!

# output nvidia interface
while ps -p $PID > /dev/null;
do
nvidia-smi > smi_0.txt
sleep 10
done