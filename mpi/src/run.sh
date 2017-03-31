# !/bin/bash

ulimit -s 65532

make

echo "Threads: $1"
mpiexec -n $1 ./ScanSky_mpi $2
