gpuErrorCheck(cudaMemsetAsync(flagCambio_d, 0, sizeof(char) * 4, stream[0]));
for (i = 0; i < nStreams; i++){
    kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu,0,stream[0]>>>(
        &matrixResult_d[rows * columns * i],
        &matrixResult_d[rows * columns * ((i - 1 + nStreams) % nStreams)],
        &flagCambio_d[i]
    );
}
gpuErrorCheck(cudaPeekAtLastError());
int s = -1;
for(t=0; flagCambio != 0; t++){
    flagCambio = 0;
    s = t % nStreams;
    gpuErrorCheck(cudaMemcpyAsync(&flagCambio,&flagCambio_d[s], sizeof(char),
        cudaMemcpyDeviceToHost,stream[s]));
    gpuErrorCheck(cudaMemcpyAsync(&flagCambio_d[s],&zero, sizeof(char),
        cudaMemcpyHostToDevice,stream[s]));
    kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu,0,stream[s]>>>(
        &matrixResult_d[rows * columns * s],
        &matrixResult_d[rows * columns * ((s - 1 + nStreams) % nStreams)],
        &flagCambio_d[s]
    );
}
