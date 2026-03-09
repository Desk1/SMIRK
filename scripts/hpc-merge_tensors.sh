#!/bin/bash --login
#SBATCH --job-name=merge-tensors
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

#SBATCH --output=jobs/job_output_%j.log
#SBATCH --error=jobs/job_error_%j.log  


# activate environment
conda activate SMIRK-HPC

# run py script
python my_merge_all_tensors.py blackbox_attack_data/vggface2/inception_resnetv1_vggface2/celeba_partial256/ &
PID=$!

# output nvidia interface
while ps -p $PID > /dev/null;
do
nvidia-smi > smi_0.txt
sleep 10
done