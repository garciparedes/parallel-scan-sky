__global__ void kernelCountFigures(int *matrixResult) {
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
 	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i<rows_d-1 &&
		j > 0 && j<columns_d-1 &&
			matrixResult[i*columns_d+j] == i*columns_d+j) {
				atomicAdd(&numBlocks_d, 1);
	}
}
