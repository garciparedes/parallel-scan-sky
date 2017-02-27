# !/bin/bash

ulimit -s 65532

make

export OMP_NUM_THREADS=4
echo "Threads: $OMP_NUM_THREADS"
./ScanSky_openmp $1

export OMP_NUM_THREADS=1
echo "Threads: $OMP_NUM_THREADS"
./ScanSky_openmp $1
