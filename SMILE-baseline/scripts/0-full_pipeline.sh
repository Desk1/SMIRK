#!/bin/bash --login
#SBATCH --job-name=MI-attack
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

# sample
# PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_sample_z_w_space.py 

# query target model
# PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_generate_blackbox_attack_dataset.py --arch_name inception_resnetv1_vggface2 vggface2 celeba_partial256

# merge
# PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_merge_all_tensors.py blackbox_attack_data/vggface2/inception_resnetv1_vggface2/celeba_partial256/ 

# train surrogate
PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python long-tailed_surrogate_training.py --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --arch_name_finetune inception_resnetv1_casia --finetune_mode 'vggface2->CASIA' --epoch 750 --batch_size 128 --query_num 2500 

# run attacks
for target in {1..10}
do
    PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_whitebox_attacks.py --attack_mode ours-w --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --epochs 400 --arch_name_finetune inception_resnetv1_casia --finetune_mode 'vggface2->CASIA' --num_experts 3 --EorOG SMILE --population_size 2500
    PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)" python my_blackbox_attacks.py --attack_mode ours-surrogate_model --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --budget 2500 --population_size 2500 --epochs 400 --finetune_mode 'vggface2->CASIA' --arch_name_finetune inception_resnetv1_casia --EorOG SMILE --lr 0.2 --x 1.7
done

# eval
targets=$(echo {1..10} | tr ' ' ',')
python my_blackbox_attacks.py --attack_mode ours-surrogate_model --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --budget 2500 --population_size 2500 --finetune_mode 'vggface2->CASIA' --arch_name_finetune inception_resnetv1_casia --EorOG SMILE --epochs 400 --lr 0.2 --test_only

