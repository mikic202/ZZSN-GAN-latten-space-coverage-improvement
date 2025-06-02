#!/bin/bash
module load GCCcore/12.3.0 Python/3.11.3
module load GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0
# module load GCC/12.3.0 matplotlib/3.7.2

echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

INDEX=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p task_$INDEX/models

python3 test_coverage.py $INDEX