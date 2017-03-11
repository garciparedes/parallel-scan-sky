/* 4.2.1 Actualizacion copia */
#pragma omp for \
schedule(static), \
private(k)
for(k=0;k<k_max;k++){
    matrixResultCopy[k_indexer[k]]=matrixResult[k_indexer[k]];
}
