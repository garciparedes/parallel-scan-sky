/* 4.2.2 Computo y detecto si ha habido cambios */
for(i=1;i<rows-1;i++){
    for(j=1;j<columns-1;j++){
        flagCambio= flagCambio+ computation(i,j,columns, matrixData, matrixResult, matrixResultCopy);
    }
}
