#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=DENOISING
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --array=0-26
#SBATCH --output=%x_%A_%a.out

# ============================================================================
# Slurm script for distributed DDPM-MCMC sample generation.
#
# 3 sigmas x 3 inits x 3 mcmcs = 27 combinations  ->  array 0-26
#
# This script is written for an HPC cluster with H100 GPUs and a
# CUDA + cuDNN module system. The exact module names, account name, and
# library paths below depend on your specific cluster — adjust them to
# match your setup. Lines marked with `# CLUSTER-SPECIFIC` are the most
# likely candidates to need editing.
# ============================================================================

set -e

# ── Cluster modules (CLUSTER-SPECIFIC: adjust to your environment) ──────────
module purge
module load arch/h100
module load python/3.11.5
module load cuda/12.1.0
module load cudnn/8.9.7.29-cuda

# ── Install the package in editable mode (no deps, no build isolation) ──────
pip install -e . --user --force-reinstall --no-deps --no-build-isolation

# ── Environment (CLUSTER-SPECIFIC: $WORK is the user scratch on this site) ──
# If your cluster does not define $WORK, set the path to your local Python
# libs directory manually, or remove these exports if you don't rely on
# pip-installed CUDA wheels.
export PYTHONPATH=$WORK/python_libs:$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=$WORK/python_libs/nvidia/cudnn/lib:$WORK/python_libs/nvidia/cublas/lib:$WORK/python_libs/nvidia/cusparse/lib:$WORK/python_libs/nvidia/cusolver/lib:$WORK/python_libs/nvidia/cufft/lib:$WORK/python_libs/nvidia/cuda_runtime/lib:$WORK/python_libs/nvidia/cuda_cupti/lib:$WORK/python_libs/nvidia/cuda_nvrtc/lib:$WORK/python_libs/nvidia/nvjitlink/lib:$WORK/python_libs/nvidia/nccl/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ── Task grid ────────────────────────────────────────────────────────────────
SIGMAS=(1.24404052 1.05527906 0.801241500)
INITS=("p_t" "gaussian" "p_theta")
MCMCS=("hmc" "barker" "nuts")

# 3 sigmas x (3 inits x 3 mcmcs) = 3 x 9 = 27
N_INNER=9  # 3 inits * 3 mcmcs

I_SIGMA=$(( SLURM_ARRAY_TASK_ID / N_INNER ))
I_REMAINDER=$(( SLURM_ARRAY_TASK_ID % N_INNER ))
I_INIT=$(( I_REMAINDER / 3 ))
I_MCMC=$(( I_REMAINDER % 3 ))

SIGMA=${SIGMAS[$I_SIGMA]}
CURRENT_INIT=${INITS[$I_INIT]}
CURRENT_MCMC=${MCMCS[$I_MCMC]}

echo "Task $SLURM_ARRAY_TASK_ID: Sigma=$SIGMA, Init=$CURRENT_INIT, MCMC=$CURRENT_MCMC"

python -u diffusion_generation_mcmc.py "$CURRENT_INIT" "$SIGMA" "$CURRENT_MCMC"

echo "Task $SLURM_ARRAY_TASK_ID completed."