#!/bin/bash

####### select partition (check CCR documentation)
#SBATCH --partition=gpu --qos=gpu

####### set memory that nodes provide (check CCR documentation, e.g., 32GB)
#SBATCH --mem=32000

####### make sure no other jobs are assigned to your nodes
#SBATCH --exclusive

####### further customizations
#SBATCH --job-name="a3"
#SBATCH --gres=gpu:2
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:02:00

####### check modules to see which version of MPI is available
####### and use appropriate module if needed
module load cuda/10.1
./a3 1000000 .002
