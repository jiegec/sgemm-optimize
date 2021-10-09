#!/bin/bash
make
echo "Running $1"
srun -n1 -w kunpeng-node108 --exclusive perf stat -e L1-dcache-loads,L1-dcache-load-misses,cycles,instructions ./$1 2>&1 | tee $1.log