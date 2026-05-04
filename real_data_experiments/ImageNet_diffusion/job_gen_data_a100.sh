#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH --job-name=GEN_FINAL
#SBATCH --partition=YOUR_PARTITION   # CLUSTER-SPECIFIC
#SBATCH --constraint=h100            # CLUSTER-SPECIFIC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-2                  # 3 seeds: array indices 0, 1, 2
#SBATCH --output=%x_%A_%a.out        # %A = job ID, %a = array index
#SBATCH --error=%x_%A_%a.err

# ============================================================================
# Slurm script for distributed data generation across multiple seeds.
#
# Each array task runs `generate_data.py` with a different seed, so the three
# tasks produce three independent generated datasets in parallel.
#
# Written for an HPC cluster with H100 GPUs and a module-based environment
# (PyTorch + CUDA pre-installed via modules). Adjust account, partition,
# constraint and module names to match your cluster.
# ============================================================================

set -x   # Print each command to the log for debugging

# ── Cluster modules (CLUSTER-SPECIFIC) ──────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.1

# ── Environment ─────────────────────────────────────────────────────────────
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# ── Seed mapping ────────────────────────────────────────────────────────────
# Array index (0, 1, 2) maps to the corresponding seed.
SEEDS=(0 1 2)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Directory: $(pwd)"
echo "Starting job with seed: $CURRENT_SEED"

# ── Execution ───────────────────────────────────────────────────────────────
# $CURRENT_SEED is passed as sys.argv[1] to the Python script.
python -u generate_data.py $CURRENT_SEED