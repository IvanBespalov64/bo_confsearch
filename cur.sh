#!/usr/bin/bash

#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=cs_orca
export KMP_STACKSIZE=10G
export OMP_STACKSIZE=10G
export OMP_NUM_THREADS=8,1
export MKL_NUM_THREADS=8

/opt/orca5/orca tests/cur.inp > tests/cur.out
