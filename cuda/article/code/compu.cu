gpuErrorCheck(cudaMemcpyAsync(flagCambio_d_0,&zero, sizeof(char),
    cudaMemcpyHostToDevice,stream[0]));
kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu,0,stream[0]>>>(
    matrixResult_d_0,matrixResult_d_3,flagCambio_d_0);
// ...
for(t=0; flagCambio != 0; t++){
    flagCambio = 0;
    if(t % nStreams == 0){
        gpuErrorCheck(cudaMemcpyAsync(&flagCambio,flagCambio_d_0, sizeof(char),
            cudaMemcpyDeviceToHost,stream[0]));
        gpuErrorCheck(cudaMemcpyAsync(flagCambio_d_0,&zero, sizeof(char),
            cudaMemcpyHostToDevice,stream[0]));
        kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu,0,stream[0]>>>(
            matrixResult_d_0,matrixResult_d_3,flagCambio_d_0);
    } else if(t % nStreams == 1){
        // ...
    } // ...
}
