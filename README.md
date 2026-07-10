# Great Lakes DDP Tutorial

A minimal tutorial for running **Distributed Data Parallel (DDP)** PyTorch training on the University of Michigan [Great Lakes](https://documentation.its.umich.edu/arc-hpc/greatlakes/user-guide) cluster.

This demo trains a small CNN on MNIST using **2 GPUs on a single node** in the `spgpu` partition.

## What is in this repo

| File | Description |
|------|-------------|
| [`train_ddp.py`](train_ddp.py) | Minimal DDP training script (MNIST + small CNN) |
| [`scripts/train_ddp.sbatch`](scripts/train_ddp.sbatch) | SLURM batch script for Great Lakes |
| [`environment.yml`](environment.yml) | Conda/mamba environment with PyTorch + CUDA 12.1 |

## Prerequisites

- A Great Lakes account with Duo MFA enabled
- A SLURM account (billing allocation) — find yours with `accountspayable` or check the [Great Lakes user guide](https://documentation.its.umich.edu/arc-hpc/greatlakes/user-guide)
- SSH access: `ssh <uniqname>@greatlakes.arc-ts.umich.edu`

## Setup (one time)

### 1. Clone the repo

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
git clone https://github.com/behradrabiei/greatlake_training_tutorial.git
cd greatlake_training_tutorial
```

You can also clone to `/scratch/<your_account>` for faster I/O during training.

### 2. Create the Python environment

```bash
module load mamba/py3.12
mamba env create -f environment.yml
```

This creates a `ddp-demo` environment with PyTorch, torchvision, and CUDA 12.1 support.

### 3. Verify GPU access (optional but recommended)

Request an interactive GPU session:

```bash
salloc --account=YOUR_SLURM_ACCOUNT \
  --partition=spgpu \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=48G \
  --time=00:30:00
```

Then activate the environment and check CUDA:

```bash
module load mamba/py3.12
source activate ddp-demo
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Exit the interactive session when done: `exit`

## Submit the job

### 1. Edit the sbatch script

Open [`scripts/train_ddp.sbatch`](scripts/train_ddp.sbatch) and replace `YOUR_SLURM_ACCOUNT` with your actual SLURM account name:

```bash
#SBATCH --account=your_account_name
```

### 2. Submit

From the repo root:

```bash
sbatch scripts/train_ddp.sbatch
```

SLURM will print a job ID, e.g. `Submitted batch job 12345678`.

### 3. Monitor

```bash
squeue -u $USER
tail -f logs/ddp-<jobid>.out
```

## Expected output

When the job succeeds, the log should show something like:

```
Starting DDP training: world_size=2, rank=0, local_rank=0
Epoch 1/2 - loss: 0.2341
Epoch 2/2 - loss: 0.0892
Training complete.
```

Key things to verify:

- `world_size=2` confirms both GPUs are participating
- Loss decreases across epochs
- Job completes in a few minutes

## How it works

1. **SLURM** allocates 1 node with 2 GPUs via `scripts/train_ddp.sbatch`
2. **`srun`** launches the job on the allocated compute node
3. **`torchrun`** starts 2 processes (one per GPU) and sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`
4. **`train_ddp.py`** initializes NCCL, wraps the model in `DistributedDataParallel`, and uses `DistributedSampler` so each GPU sees a different shard of MNIST

```
Login node  --sbatch-->  SLURM  -->  spgpu node
                                         |
                                    srun + torchrun
                                    /            \
                               GPU 0             GPU 1
                               (rank 0)          (rank 1)
                                    \            /
                                     NCCL sync
```

## Local testing (optional)

If you have a machine with 2+ GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py
```

## Customization

Edit constants at the top of [`train_ddp.py`](train_ddp.py):

- `EPOCHS` — number of training epochs (default: 2)
- `BATCH_SIZE` — per-GPU batch size (default: 64)
- `NUM_WORKERS` — DataLoader workers per process (default: 2)

To use more GPUs on a single node, update both the sbatch script (`--ntasks-per-node`, `--gres=gpu:N`, `--mem`) and ensure `nproc_per_node` matches.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Job rejected with `QOSMinGRES` | You submitted to a GPU partition without requesting GPUs. Ensure `--gres=gpu:2` is set. |
| Job rejected with `InvalidAccount` | Replace `YOUR_SLURM_ACCOUNT` with your real account name. |
| `CUDA available: False` in interactive test | Make sure you requested a GPU with `salloc --gres=gpu:1` and are on a compute node, not a login node. |
| `ModuleNotFoundError: torch` | Run `module load mamba/py3.12 && source activate ddp-demo` before running Python. |
| NCCL / distributed errors | Confirm `--ntasks-per-node` matches `--gres=gpu:N` and `nproc_per_node`. |

For more help, see the [Slurm user guide](https://documentation.its.umich.edu/arc-hpc/slurm-user-guide) and [PyTorch on Great Lakes](https://documentation.its.umich.edu/arc-software/pytorch).

## Next steps

This tutorial covers **single-node, multi-GPU** DDP. Multi-node training requires additional setup (`MASTER_ADDR`, `MASTER_PORT`, `--nodes>1`) and is not included here.
