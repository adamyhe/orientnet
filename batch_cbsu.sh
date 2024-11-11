#!/bin/bash -l
#SBATCH --job-name=orientnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=1-9

# Script to run the training of the model on the CBSU cluster

echo "Starting job $SLURM_ARRAY_TASK_ID on $SLURM_JOB_NODELIST"

# Set up the environment

echo "Setting up environment"
conda activate clipnet

# Run the training script

echo "Running training script"

cd /home2/ayh8/orientnet/
time python transfer_learn_orientation.py $SLURM_ARRAY_TASK_ID 0