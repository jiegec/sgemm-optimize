#!/bin/bash
echo "Running $1"
srun -n1 --exclusive ./$1 | tee $1.log