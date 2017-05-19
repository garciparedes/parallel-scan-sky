kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(
    &matrixResult_d[rows * columns * 0]
);
gpuErrorCheck(cudaPeekAtLastError());
