gpuErrorCheck(cudaMalloc(&flagCambio_d_0, sizeof(char)));
// ...
gpuErrorCheck(cudaMallocPitch(&matrixResult_d_0, &pitch, rows*sizeof(int), columns));
// ...
gpuErrorCheck(cudaMallocPitch(&matrixData_d, &pitch3, rows*sizeof(char), columns));
gpuErrorCheck(cudaMemcpyToSymbolAsync(rows_d,&rows, sizeof(int),0,cudaMemcpyHostToDevice));
gpuErrorCheck(cudaMemcpyToSymbolAsync(columns_d,&columns, sizeof(int),0,cudaMemcpyHostToDevice));
gpuErrorCheck(cudaMemcpyToSymbolAsync(matrixData_d,&matrixDataChar_d, sizeof(char *)));
matrixDataChar = (char *)malloc( rows * columns * sizeof(char) );
for(i = 0; i < rows * columns; i++){
    matrixDataChar[i] = matrixData[i];
}
gpuErrorCheck(cudaMemcpyAsync(matrixDataChar_d,matrixDataChar,
    sizeof(char) * rows * columns, cudaMemcpyHostToDevice));
