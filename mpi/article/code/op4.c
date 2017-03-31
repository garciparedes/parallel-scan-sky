if (world_size != 1) {
    if(t !=  0){
        MPI_Wait(&request[4], MPI_STATUS_IGNORE);
    }
    if (world_rank != world_size - 1) {
        MPI_Wait(&request[2], MPI_STATUS_IGNORE);
        MPI_Start(&request[2]);
    }
    if (world_rank != 0) {
        MPI_Wait(&request[3], MPI_STATUS_IGNORE);
        MPI_Start(&request[3]);
    }
    flagCambio=0;
    MPI_Iallreduce(&local_flagCambio, &flagCambio, 1, MPI_CHAR, MPI_LOR,
        MPI_COMM_WORLD, &request[4]);
    if (!local_flagCambio){
        MPI_Wait(&request[4], MPI_STATUS_IGNORE);
    }
}
