# !/bin/bash

ulimit -s 65532

make

./ScanSky_mpi $1
