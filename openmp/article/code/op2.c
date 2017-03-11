/* 4.2.1 Actualizacion copia */
#pragma omp single
{
    flagCambio=0;
    temp = matrixResultCopy;
    matrixResultCopy = matrixResult;
    matrixResult = temp;
}
