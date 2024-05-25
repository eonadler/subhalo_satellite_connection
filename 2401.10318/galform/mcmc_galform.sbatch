#!/bin/bash -l
#SBATCH --partition=any
#SBATCH -t 168:00:00
#SBATCH --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH -J dwarf_forecast_cdm_galform
#SBATCH --output=dwarf_forecast_cdm_galform.%j.oe
#SBATCH --mem-per-cpu=32G

module load mpich/4.0.0

mpirun -n $SLURM_NTASKS /central/software/python/3.7.0/bin/python3.7 mcmc_galform.py
