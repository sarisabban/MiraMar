#!/bin/sh
#SBATCH --job-name=MolTet
#SBATCH --partition=compsci
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

cd $SLURM_SUBMIT_DIR

python3 MolecularTetris.py -rl
