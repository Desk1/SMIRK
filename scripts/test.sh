#!/bin/bash --login
#SBATCH --job-name=test_script # Replace with the name of your job
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk # use your own email
#SBATCH --mail-type=BEGIN,END,FAIL # when to send notification email
#SBATCH -o slurm.%j.out # STDOUT '%j' will be replaced with job-id
#SBATCH -e slurn.%j.err # STDERR
#SBATCH --ntasks=1 # The number of tasks
# (default is 1 cpu per task)
#SBATCH --time=00:05:00 # length of time your job needs
#SBATCH --partition=compute # 'queue' job to be submitted to

module load conda
conda activate SMIRK-HPC
pwd
