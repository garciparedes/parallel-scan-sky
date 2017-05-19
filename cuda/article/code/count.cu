numBlocks = 0;
gpuErrorCheck(cudaMemcpyToSymbolAsync(numBlocks_d,&numBlocks,
    sizeof(int),0,cudaMemcpyHostToDevice));
if(t % nStreams == 0){
    kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>(matrixResult_d_0);
} else if(t % nStreams == 1){
    // ...
} // ...
gpuErrorCheck(cudaPeekAtLastError());
gpuErrorCheck(cudaMemcpyFromSymbolAsync(&numBlocks, numBlocks_d,
    sizeof(int), 0, cudaMemcpyDeviceToHost));
