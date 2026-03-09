#!/bin/bash --login
#SBATCH --job-name=unzip-vggface2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.jacob.22@abdn.ac.uk
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=1:00:00

#SBATCH --output=jobs/job_output_%j.log
#SBATCH --error=jobs/job_error_%j.log  


unzip datasets/vggface2.zip