#!/bin/bash
#SBATCH -A trq@a100
#SBATCH --job-name=GenFlow_Array
#SBATCH --partition=gpu_p5           # Keeping your original partition
#SBATCH --constraint=a100            # Keeping your original constraint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=gen_%A_%a.out       # %A=JobID, %a=ArrayIndex
#SBATCH --error=gen_%A_%a.err
#SBATCH --array=0-11                 # 12 total jobs (3 Seeds x 4 CFGs)

# --- Load Modules ---
module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0

# --- Environment Variables ---
export PYTHONUNBUFFERED=1

# Move to project directory

# --- Parameter Mapping Logic ---
# Define your target values
SEEDS=(0 1 2)
CFGS=(0.5 1.0 1.5 2.0)
DYNAMICAL_VAL=${1:-0}

# Map the SLURM_ARRAY_TASK_ID to the array indices
# This math ensures every combination is covered
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))
CFG_IDX=$((SLURM_ARRAY_TASK_ID % 4))

CURRENT_SEED=${SEEDS[$SEED_IDX]}
CURRENT_CFG=${CFGS[$CFG_IDX]}

echo "------------------------------------------------"
echo "🚀 Starting Generation"
echo "Job Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Selected Seed: $CURRENT_SEED"
echo "Selected CFG:  $CURRENT_CFG"
echo "📅 Date:        $(date)"
echo "------------------------------------------------"

# Execution:
# sys.argv[1] -> dynamical
# sys.argv[2] -> SEED
# sys.argv[3] -> CFG
# Note: Ensure Python uses seed = int(sys.argv[1])
python generating_noised_lightning.py $DYNAMICAL_VAL $CURRENT_SEED $CURRENT_CFG

echo "✅ Generation completed at: $(date)"