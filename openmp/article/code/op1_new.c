/* 4.2.2 Computo y detecto si ha habido cambios */
#pragma omp for \
schedule(static), \
reduction(||:flagCambio),\
private(k)
for(k=0;k<k_max;k++){
    if((matrixData[k_indexer[k]-columns] == matrixData[k_indexer[k]]) &&
        (matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]-columns]))
    {
        matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]-columns];
        flagCambio = 1;
    }
    if((matrixData[k_indexer[k]+columns] == matrixData[k_indexer[k]]) &&
        (matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]+columns]))
    {
        matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]+columns];
        flagCambio = 1;
    }
    if((matrixData[k_indexer[k]-1] == matrixData[k_indexer[k]]) &&
        (matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]-1]))
    {
        matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]-1];
        flagCambio = 1;
    }
    if((matrixData[k_indexer[k]+1] == matrixData[k_indexer[k]]) &&
        (matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]+1]))
    {
        matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]+1];
        flagCambio = 1;
    }
}
