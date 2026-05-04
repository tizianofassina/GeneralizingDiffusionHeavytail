#!/bin/bash
#SBATCH -A trq@a100                # Project account for A100 resources
#SBATCH --job-name=TarFlow_Dyn     # Job name
#SBATCH --partition=gpu_p5         # A100 80GB partition
#SBATCH --constraint=a100          # Hardware constraint
#SBATCH --nodes=1                  # Total number of nodes
#SBATCH --ntasks-per-node=4        # 1 task per GPU (Standard DDP setup)
#SBATCH --gres=gpu:4               # Total number of GPUs reserved
#SBATCH --cpus-per-task=8          # 8 CPU cores per GPU
#SBATCH --hint=nomultithread       # Disable hyperthreading
#SBATCH --time=08:00:00            # Max walltime (HH:MM:SS)
#SBATCH --output=slurm-%j.out      # Standard output file (%j expands to JobID)
#SBATCH --error=slurm-%j.err       # Standard error file

# 1. MODULE LOADING
module purge
module load arch/a100              # Critical for A100 node architecture
module load pytorch-gpu/py3/2.3.0  # Pre-configured PyTorch environment

# 2. ENVIRONMENT VARIABLES
export PYTHONUNBUFFERED=1          # Ensure logs are printed in real-time
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. DYNAMICAL PARAMETER MANAGEMENT
# Capture the first argument passed to 'sbatch'. Defaults to "false" if empty.
DYNAMICAL_VAL=${1:-0}
echo "------------------------------------------------"
echo "Job started at: $(date)"
echo "Dynamical noise setting: $DYNAMICAL_VAL"
echo "Running on node: $SLURMD_NODENAME"
echo "------------------------------------------------"

# Move to the project directory
# Make sure $REPLICA (or your specific path) is correctly defined on the cluster

# Check if the training script exists before execution
if [ ! -f train_noised_lightning.py ]; then
    echo "ERROR: train_noised_lightning.py not found in $(pwd)"
    exit 1
fi

# 4. EXECUTION
# srun launches 4 processes (1 per GPU). 
# The $DYNAMICAL variable is passed as the first command-line argument.
srun python -u train_noised_lightning.py $DYNAMICAL_VAL
