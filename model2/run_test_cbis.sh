#!/bin/bash

#SBATCH --job-name=test-cbis    # Job name
#SBATCH --output=test-cbis.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu               # Schedule a GPU
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown    # Run on either the red or brown queue

module purge

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/1.15.2-fosscuda-2019b-Python-3.7.4

VENVNAME="venv1"

source $HOME/venvs/$VENVNAME/bin/activate

python test_cbis.py