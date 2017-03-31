local_numBlocks = 0;
for(i=row_init;i<row_end;i++){
	const int t1 = (i+row_shift*world_rank)*rows_columns[1];
	for(j=1;j<rows_columns[1]-1;j++){
		if(matrixResult[i*rows_columns[1]+j] == t1+j) local_numBlocks++;
	}
}
MPI_Reduce(&local_numBlocks, &numBlocks, 1, MPI_INT, MPI_SUM, 0,
   MPI_COMM_WORLD);
