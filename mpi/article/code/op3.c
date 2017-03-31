if (world_size != 1 && t != 0){
    if (world_rank == world_size -1 ) {
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        MPI_Start(&request[1]);
        if (!local_flagCambio){
            // ...
        }
    } else if (world_rank == 0 ) {
        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        MPI_Start(&request[0]);
        if (!local_flagCambio){
            //...
        }
    } else {
        MPI_Waitall(2, request, MPI_STATUS_IGNORE);
        MPI_Startall(2,request);
        if (!local_flagCambio){
            for(j = 1; j < rows_columns[1]-1;j++){
                if (matrixResult[(row_init-1)*rows_columns[1]+j]!=
                    matrixResultCopy[(row_init-1)*rows_columns[1]+j]) {
                    local_flagCambio = 1;
                    break;
                }
                if (matrixResult[(row_end)*rows_columns[1]+j]!=
                    matrixResultCopy[(row_end)*rows_columns[1]+j]) {
                    local_flagCambio = 1;
                    break;
                }
            }
        }
    }
}
