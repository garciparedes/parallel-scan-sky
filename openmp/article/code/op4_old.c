/* 4.3 Inicio cuenta del numero de bloques */
numBlocks=0;
for(i=1;i<rows-1;i++){
    for(j=1;j<columns-1;j++){
        if(matrixResult[i*columns+j] == i*columns+j) numBlocks++;
    }
}
