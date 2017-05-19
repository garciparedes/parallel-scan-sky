kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(
    &matrixResult_d[rows * columns * (nStreams-1)]
);
gpuErrorCheck(cudaPeekAtLastError());
