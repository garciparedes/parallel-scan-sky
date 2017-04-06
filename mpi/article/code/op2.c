if (world_rank == 0){
    MPI_Type_vector((row_shift + 2)*rows_columns[1], 1,
        sizeof(int)/sizeof(char), MPI_CHAR, &data_type );
    MPI_Type_commit(&data_type);
    for(i = 1; i < world_size-1; i++ ){
        MPI_Isend(&matrixData[(row_shift*i)*(rows_columns[1])],1,
            data_type, i, 0, MPI_COMM_WORLD, &request[0]);
    }
    MPI_Type_free(&data_type);
    MPI_Type_vector(((rows_columns[0]) - (row_shift*(world_size-1)))*rows_columns[1],
        1,sizeof(int)/sizeof(char), MPI_CHAR, &data_type );
    MPI_Type_commit(&data_type);
    MPI_Isend(&matrixData[(row_shift*(world_size-1))*(rows_columns[1])],1,
        data_type, world_size-1, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Type_free(&data_type);
} else {
    MPI_Type_vector((2 + row_end - row_init)*rows_columns[1],1,
        sizeof(int)/sizeof(char), MPI_CHAR, &data_type );
    MPI_Type_commit(&data_type);
    MPI_Irecv(matrixData, 1, data_type, 0, 0, MPI_COMM_WORLD, &request[4]);
    MPI_Type_free(&data_type);
    MPI_Recv_init(&matrixResultCopy[(row_init-1)*rows_columns[1]+1], 1,
        row_type, world_left, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Send_init(&matrixResultCopy[(row_init)*rows_columns[1]+1], 1,
        row_type, world_left, 0, MPI_COMM_WORLD, &request[3]);
    MPI_Start(&request[1]);
    MPI_Wait(&request[4], MPI_STATUS_IGNORE);
}
if (world_rank != world_size - 1) {
    MPI_Recv_init(&matrixResultCopy[(row_end)*rows_columns[1]+1], 1,
        row_type, world_right, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Send_init(&matrixResultCopy[(row_end-1)*rows_columns[1]+1], 1,
        row_type, world_right, 0, MPI_COMM_WORLD, &request[2]);
    MPI_Start(&request[0]);
}
