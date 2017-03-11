for(i=0;i< rows; i++){
    for(j=0;j< columns; j++){
        matrixResult[i*(columns)+j]=-1;
        // Si es 0 se trata del fondo y no lo computamos
        if(matrixData[i*(columns)+j]!=0){
            matrixResult[i*(columns)+j]=i*(columns)+j;
        }
    }
}
