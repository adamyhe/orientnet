#!/bin/bash -l
#SBATCH --job-name=attr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=1-9

# Script to run attribution of the model on the CBSU cluster
echo "Starting job $SLURM_ARRAY_TASK_ID on $SLURM_JOB_NODELIST"

# Set up the environment
echo "Setting up environment"
conda activate clipnet

# Run the attribution script
echo "Running attribution script"
cd /home2/ayh8/orientnet/
time python calculate_deepshap.py \
    ../data/lcl/all_tss_windows_reference_seq.fna.gz \
    ../data/lcl/all_tss_windows_deepshap_${SLURM_ARRAY_TASK_ID}.npz \
    ../data/lcl/all_tss_windows_ohe.npz \
    --model_fp ensemble_models_logits_rescale_true/fold_${SLURM_ARRAY_TASK_ID}.h5 \
    --gpu 0

time python calculate_deepshap.py \
    ../data/lcl/all_tss_windows_reference_seq.fna.gz \
    ../data/lcl/all_tss_windows_deepshap_9.npz \
    ../data/lcl/all_tss_windows_ohe.npz \
    --model_fp ensemble_models_logits_rescale_true/fold_9.h5 \
    --gpu 1