/* 4.3 Inicio cuenta del numero de bloques */
#pragma omp for \
schedule(dynamic, k_max/omp_get_num_threads()), \
private(k),\
reduction(+:numBlocks)
for(k=0;k<k_max;k++){
    if(matrixResult[k_indexer[k]] == k_indexer[k]) numBlocks++;
}
