#!/bin/bash
#SBATCH -A trq@a100                # Project account
#SBATCH --job-name=AggData_300     # Job name
#SBATCH --partition=gpu_p5         # Use A100 partition for high RAM availability
#SBATCH --constraint=a100          # Hardware constraint
#SBATCH --nodes=1                  # Single node is sufficient
#SBATCH --ntasks=1                 # Single task (Data processing is usually serial)
#SBATCH --gres=gpu:1               # Reserve 1 GPU to access specific node resources
#SBATCH --cpus-per-task=16         # Increase cores for faster .npy I/O processing
#SBATCH --hint=nomultithread       # Disable hyperthreading for stability
#SBATCH --time=10:00:00            # 2 hours (estimated for 300 classes)
#SBATCH --output=agg_data-%j.out   # Standard output log
#SBATCH --error=agg_data-%j.err    # Error log

# 1. MODULE LOADING
module purge
module load arch/a100              # Prevents instruction set errors
module load pytorch-gpu/py3/2.3.0  # Standard PyTorch environment

# 2. ENVIRONMENT VARIABLES
export PYTHONUNBUFFERED=1          # Real-time logging
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. EXECUTION
# Ensure you are in the directory containing your script and JSON
# cd /path/to/your/work/directory

echo "Starting data aggregation at: $(date)"

# Use 'python -u' to ensure logs are written to the .out file immediately
python -u data_preprocess.py

echo "Aggregation finished at: $(date)"