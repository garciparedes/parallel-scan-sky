#pragma omp parallel \
default(none), \
shared(k_indexer, k_max, matrixData, matrixResult, \
    matrixResultCopy,columns,rows, flagCambio,numBlocks)
{
    // ...
}
