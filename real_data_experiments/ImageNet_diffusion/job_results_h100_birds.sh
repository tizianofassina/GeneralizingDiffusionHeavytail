#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=metrics_birds_h100
#SBATCH --partition=YOUR_PARTITION   # CLUSTER-SPECIFIC
#SBATCH --constraint=h100            # CLUSTER-SPECIFIC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================================
# Slurm script for computing evaluation metrics on the birds dataset (H100).
#
# Written for an HPC cluster with H100 GPUs and a module-based environment
# (PyTorch + CUDA pre-installed via modules). Adjust account, partition,
# constraint and module names to match your cluster.
# ============================================================================

set -x

# ── Cluster modules (CLUSTER-SPECIFIC) ──────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.1

# ── Environment ─────────────────────────────────────────────────────────────
# Force unbuffered output so OOM errors and other failures show up
# immediately in the log.
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Launching metrics computation on H100 (birds)..."

# ── Execution ───────────────────────────────────────────────────────────────
python -u compute_results_birds.py