#!/bin/bash --login
#SBATCH --job-name=eval-attack
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

# eval
targets=$(echo {1..10} | tr ' ' ',')
python my_blackbox_attacks.py --attack_mode ours-surrogate_model --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --budget 2500 --population_size 2500 --finetune_mode 'vggface2->CASIA' --arch_name_finetune inception_resnetv1_casia --EorOG SMILE --epochs 400 --lr 0.2 --test_only