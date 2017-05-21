# !/bin/bash

ulimit -s 65532

nvcc -O3 -arch=sm_35 -o ScanSky_cuda ScanSky_cuda.cu

./ScanSky_cuda $1
