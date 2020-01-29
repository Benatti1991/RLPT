#!/bin/bash
#SBATCH --job-name=RunOptixSample
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=21
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#
# Edit next line
#SBATCH -- defaultaccount g_chrono
#

eval "$(conda shell.bash hook)"

conda activate sensor

export PYTHONPATH=/hpc/home/simone.benatti/common/chrono/chrono-sens/chrono_build/bin/

python ./ppo.py





