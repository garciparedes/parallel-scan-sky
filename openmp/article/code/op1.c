/* 3. Etiquetado inicial */
#pragma omp for \
nowait,\
schedule(dynamic, ((rows-1)*(columns-1))/omp_get_num_threads()), \
private(i,j,k)
for(i = 1; i < rows-1; i++){
    for(j = 1; j < columns-1; j++){
        // Si es 0 se trata del fondo y no lo computamos
        if(matrixData[i*(columns)+j]!=0){
            matrixResult[i*(columns)+j]=i*(columns)+j;
            #pragma omp atomic capture
            {
                k = k_max; k_max++;
            }
            k_indexer[k] = i*(columns)+j;
        } else {
            matrixResult[i*(columns)+j]=-1;
        }
    }
    matrixResult[i*(columns)]=-1;
    matrixResult[i*(columns)+columns-1]=-1;
}

#pragma omp for \
nowait,\
schedule(static), \
private(j)
for(j=1;j< columns-1; j++){
    matrixResult[j]=-1;
    matrixResult[(rows-1)*(columns)+j]=-1;
}
