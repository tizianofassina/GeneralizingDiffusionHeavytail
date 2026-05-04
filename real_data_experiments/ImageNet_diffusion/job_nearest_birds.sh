#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=NN_ANALYSIS
#SBATCH --partition=YOUR_PARTITION   # CLUSTER-SPECIFIC
#SBATCH --constraint=a100            # CLUSTER-SPECIFIC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================================
# Slurm script for nearest-neighbor qualitative analysis on the birds dataset.
#
# Written for an HPC cluster with A100 GPUs and a module-based environment
# (PyTorch + CUDA pre-installed via modules). Adjust account, partition,
# constraint and module names to match your cluster.
# ============================================================================

set -x

# ── Cluster modules (CLUSTER-SPECIFIC) ──────────────────────────────────────
module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0

# ── Environment ─────────────────────────────────────────────────────────────
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

echo "Directory: $(pwd)"
echo "Starting nearest-neighbors qualitative analysis (birds)..."

# ── Execution ───────────────────────────────────────────────────────────────
python -u nearest_for_plot_birds.py