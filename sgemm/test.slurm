#!/bin/bash

#SBATCH -J gemm
#SBATCH -o test.out
#SBATCH -p LONG
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH --exclusive

./benchmark-test
