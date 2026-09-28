#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=VESGM_TRAIN
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out

# ============================================================================
# Slurm script: VE-SGM denoiser training, one train-set size per array task.
#
# 5 sizes -> array 0-4, one process per size.
# Lines marked `# CLUSTER-SPECIFIC` are the ones to adjust to your setup.
# ============================================================================

set -e

# ── Cluster modules (CLUSTER-SPECIFIC) ──────────────────────────────────────
# PyTorch stack: the pytorch-gpu module already ships torch + CUDA + cuDNN,
# so we do NOT pip-install CUDA wheels here (see note below).
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

# ── Make the local package importable (VE_SGM). --no-deps: torch/lightning ──
# come from the module, so pip must not try to pull them (no internet on nodes)
pip install -e . --user --force-reinstall --no-deps --no-build-isolation

# ── Environment ─────────────────────────────────────────────────────────────
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONUNBUFFERED=1

# ── Task grid ────────────────────────────────────────────────────────────────
SIZES=(1000 5000 10000 50000 100000)

N=${SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: train_size=$N"

python -u training_different_denoiser_dim_10.py --train-size $N

echo "Task $SLURM_ARRAY_TASK_ID completed."