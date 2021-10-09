#!/bin/bash
make
echo "Running $1"
srun -n1 --exclusive perf record -o perf.data ./$1