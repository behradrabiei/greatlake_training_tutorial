#!/bin/bash
#SBATCH --job-name=ddp-demo
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ddp-%j.out

set -euo pipefail

module load mamba/py3.12
source activate ddp-demo

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  train_ddp.py
