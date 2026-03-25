#!/bin/bash --login
#SBATCH --job-name=surrogate-training
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

#SBATCH --output=jobs/job_output_%j.log
#SBATCH --error=jobs/job_error_%j.log  


# activate environment
conda activate SMIRK-HPC

# run py script
python long-tailed_surrogate_training.py --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --arch_name_finetune inception_resnetv1_casia --finetune_mode 'vggface2->CASIA' --epoch 200 --batch_size 128 --query_num 2500 &
PID=$!

# output nvidia interface
while ps -p $PID > /dev/null;
do
nvidia-smi > smi_0.txt
sleep 10
done