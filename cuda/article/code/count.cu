numBlocks = 0;
gpuErrorCheck(cudaMemcpyToSymbolAsync(numBlocks_d,&numBlocks,
    sizeof(int),0,cudaMemcpyHostToDevice));
kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>(
    &matrixResult_d[rows * columns * (t % nStreams)]
);
gpuErrorCheck(cudaPeekAtLastError());
gpuErrorCheck(cudaMemcpyFromSymbolAsync(&numBlocks, numBlocks_d,
    sizeof(int), 0, cudaMemcpyDeviceToHost));
