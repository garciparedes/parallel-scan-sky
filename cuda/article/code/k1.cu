__global__ void kernelFillMatrixResult(int *matrixResult) {
    const int ij = (blockIdx.y * blockDim.y + threadIdx.y)*columns_d +
        blockIdx.x * blockDim.x + threadIdx.x;
    if(ij > -1 && ij<rows_d*columns_d){
        if(matrixData_d[ij] !=0){
            matrixResult[ij]=ij;
        } else {
            matrixResult[ij]=-1;
        }
    }
}
