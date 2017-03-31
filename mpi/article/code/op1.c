MPI_Ibcast(rows_columns, 2, MPI_INT, 0, MPI_COMM_WORLD, &request[0]);
if (world_rank != 0){
    MPI_Wait(&request[0], MPI_STATUS_IGNORE);
}
MPI_Type_contiguous( rows_columns[1]-2, MPI_INT, &row_type );
MPI_Type_commit(&row_type);
row_shift = (rows_columns[0])/world_size;
row_init = 1;
row_end = row_shift +1;
if(world_rank == world_size-1){
    row_end = (rows_columns[0]-1)- (row_shift*(world_size-1));
}
