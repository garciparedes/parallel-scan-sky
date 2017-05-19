__global__ void kernelComputationLoop(int *matrixResult,int *matrixResultCopy,
    char *flagCambio_d) {
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i<rows_d-1 &&
		j > 0 && j<columns_d-1 &&
		matrixResult[i*columns_d+j] != -1){
		matrixResult[i*columns_d+j] = matrixResultCopy[i*columns_d+j];
		if((matrixData_d[(i-1)*columns_d+j] == matrixData_d[i*columns_d+j]) &&
			(matrixResult[i*columns_d+j] > matrixResultCopy[(i-1)*columns_d+j]))
		{
			matrixResult[i*columns_d+j] = matrixResultCopy[(i-1)*columns_d+j];
			*flagCambio_d = 1;
		}
		// ...
	}
}
