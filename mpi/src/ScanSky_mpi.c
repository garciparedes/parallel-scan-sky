/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código MPI
*
* @author Ana Moretón Fernández
* @author Eduardo Rodríguez Gutiez
* @author Sergio García Prado (@garciparedes)
* @author Adrian Calvo Rojo
* @version v2.3
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "cputils.h"
#include <mpi.h>


/**
* Funcion principal
*/
int main (int argc, char* argv[])
{

	/* 1. Leer argumento y declaraciones */
	if (argc < 2) 	{
		printf("Uso: %s <imagen_a_procesar>\n", argv[0]);
		return(EXIT_SUCCESS);
	}
	char* image_filename = argv[1];

	int rows=-1;
	int columns =-1;
	int *matrixData=NULL;
	int *matrixResult=NULL;
	int *matrixResultCopy=NULL;
	int numBlocks=-1;
	int world_rank = -1;
	int world_size = -1;
	double t_ini;
	int i,j;

	int *temp=NULL;

	int local_numBlocks=-1;
	int *rows_columns = (int *)malloc( 2 * sizeof(int) );

	int row_shift =-1;
	int column_shift =-1;

	int row_init =-1;
	int row_end =-1;

	int t=-1;
	int flagCambio=-1;
	int local_flagCambio=-1;

	MPI_Datatype row_type;
	MPI_Datatype data_type;
	MPI_Request *request = (MPI_Request *)malloc( 5 * sizeof(MPI_Request) );

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);

	int world_right = (world_rank +1 ) % world_size;
	int world_left = (world_rank -1 + world_size) % world_size;


	if ( world_rank == 0 ) {

		/* 2. Leer Fichero de entrada e inicializar datos */

		/* 2.1 Abrir fichero */
		FILE *f = cp_abrir_fichero(image_filename);

		// Compruebo que no ha habido errores
		if (f==NULL)
		{
			perror ("Error al abrir fichero.txt");
			return -1;
		}

		/* 2.2 Leo valores del fichero */
		int valor;
		fscanf (f, "%d\n", &rows);
		fscanf (f, "%d\n", &columns);
		// Añado dos filas y dos columnas mas para los bordes
		rows=rows+2;
		columns = columns+2;

		/* 2.3 Reservo la memoria necesaria para la matriz de datos */
		matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
		if ( (matrixData == NULL)   ) {
			perror ("Error reservando memoria");
			return -1;
		}

		/* 2.4 Inicializo matrices */
		for(i=0;i< rows; i++){
			for(j=0;j< columns; j++){
				matrixData[i*(columns)+j]=-1;
			}
		}
		/* 2.5 Relleno bordes de la matriz */
		for(i=1;i<rows-1;i++){
			matrixData[i*(columns)+0]=0;
			matrixData[i*(columns)+columns-1]=0;
		}
		for(i=1;i<columns-1;i++){
			matrixData[0*(columns)+i]=0;
			matrixData[(rows-1)*(columns)+i]=0;
		}
		/* 2.6 Relleno la matriz con los datos del fichero */
		for(i=1;i<rows-1;i++){
			for(j=1;j<columns-1;j++){
				fscanf (f, "%d\n", &matrixData[i*(columns)+j]);
			}
		}
		fclose(f);

		#ifdef WRITE
		printf("Inicializacion \n");
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixData[i*(columns)+j]);
			}
			printf("\n");
		}
		#endif


		/* PUNTO DE INICIO MEDIDA DE TIEMPO */
		t_ini = cp_Wtime();

		//
		// EL CODIGO A PARALELIZAR COMIENZA AQUI
		//
		rows_columns[0] = rows;
		rows_columns[1] = columns;
	}
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

	if (world_rank != 0){
		matrixData= (int *)malloc( (2 + row_end - row_init)*(rows_columns[1]) * sizeof(int) );
		for(i = 0; i < (2 + row_end - row_init)*(rows_columns[1]); i++){
			matrixData[i] = 0;
		}
	}
	/* 3. Etiquetado inicial */
	matrixResult= (int *)malloc( (2 + row_end - row_init)*(rows_columns[1]) * sizeof(int) );
	matrixResultCopy= (int *)malloc( (2 + row_end - row_init)*(rows_columns[1]) * sizeof(int) );
	if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
		perror ("Error reservando memoria");
		return -1;
	}
	if (world_size != 1) {

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
	}

	for(i=row_init-1;i< row_end+1; i++){
		const int t1 = (i+row_shift*world_rank)*rows_columns[1];
		for(j=0;j< rows_columns[1]; j++){
			// Si es 0 se trata del fondo y no lo computamos
			if(matrixData[i*(rows_columns[1])+j]!=0){
				matrixResult[i*(rows_columns[1])+j]=t1+j;
			} else {
				matrixResult[i*(rows_columns[1])+j]=-1;
			}
		}
	}


	/* 4. Computacion */
	t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	flagCambio=0;
	local_flagCambio=1;

	/* 4.2 Busqueda de los bloques similiares */
	for(t=0; local_flagCambio !=0 || flagCambio !=0; t++){


		/* 4.2.1 Actualizacion copia */
		if (world_size != 1 && t != 0){
			if (world_rank == world_size -1 ) {
				MPI_Wait(&request[1], MPI_STATUS_IGNORE);
				MPI_Start(&request[1]);
				if (!local_flagCambio){
					for(j = 1; j < rows_columns[1]-1;j++){
						if (matrixResult[(row_init-1)*rows_columns[1]+j] !=
							matrixResultCopy[(row_init-1)*rows_columns[1]+j]) {
							local_flagCambio = 1;
							break;
						}
					}
				}
			} else if (world_rank == 0 ) {
				MPI_Wait(&request[0], MPI_STATUS_IGNORE);
				MPI_Start(&request[0]);
				if (!local_flagCambio){
					for(j = 1; j < rows_columns[1]-1;j++){
						if (matrixResult[(row_end)*rows_columns[1]+j]
							!= matrixResultCopy[(row_end)*rows_columns[1]+j]) {
							local_flagCambio = 1;
							break;
						}
					}
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
		temp = matrixResultCopy;
		matrixResultCopy = matrixResult;
		matrixResult = temp;
		if (local_flagCambio) {
			local_flagCambio = 0;
			i = row_init;
			/* 4.2.2 Computo y detecto si ha habido cambios */
			for(j=1;j<rows_columns[1]-1;j++){
				if(matrixResult[i*rows_columns[1]+j] != -1){
					matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j];
					if((matrixData[(i-1)*rows_columns[1]+j] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[(i-1)*rows_columns[1]+j]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[(i-1)*rows_columns[1]+j];
						local_flagCambio = 1;
					}
					if((matrixData[(i+1)*rows_columns[1]+j] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[(i+1)*rows_columns[1]+j]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[(i+1)*rows_columns[1]+j];
						local_flagCambio = 1;
					}
					if((matrixData[i*rows_columns[1]+j-1] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[i*rows_columns[1]+j-1]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j-1];
						local_flagCambio = 1;
					}
					if((matrixData[i*rows_columns[1]+j+1] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[i*rows_columns[1]+j+1]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j+1];
						local_flagCambio = 1;
					}
				}
			}

			if (world_size != 1 && world_rank != 0) {
				MPI_Wait(&request[3], MPI_STATUS_IGNORE);
				MPI_Start(&request[3]);
			}
			i = row_end-1;
			for(j=1;j<rows_columns[1]-1;j++){
				if(matrixResult[i*rows_columns[1]+j] != -1){
					matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j];
					if((matrixData[(i-1)*rows_columns[1]+j] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[(i-1)*rows_columns[1]+j]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[(i-1)*rows_columns[1]+j];
						local_flagCambio = 1;
					}
					if((matrixData[(i+1)*rows_columns[1]+j] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[(i+1)*rows_columns[1]+j]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[(i+1)*rows_columns[1]+j];
						local_flagCambio = 1;
					}
					if((matrixData[i*rows_columns[1]+j-1] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[i*rows_columns[1]+j-1]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j-1];
						local_flagCambio = 1;
					}
					if((matrixData[i*rows_columns[1]+j+1] == matrixData[i*rows_columns[1]+j]) &&
						(matrixResult[i*rows_columns[1]+j] > matrixResultCopy[i*rows_columns[1]+j+1]))
					{
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j+1];
						local_flagCambio = 1;
					}
				}
			}
			if (world_size != 1 && world_rank != world_size - 1) {
				MPI_Wait(&request[2], MPI_STATUS_IGNORE);
				MPI_Start(&request[2]);
			}

			for(i=row_init+1;i<row_end-1;i++){
				for(j=1;j<rows_columns[1]-1;j++){
					if(matrixResult[i*rows_columns[1]+j] != -1){
						matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j];
						if((matrixData[(i-1)*rows_columns[1]+j] == matrixData[i*rows_columns[1]+j]) &&
						    (matrixResult[i*rows_columns[1]+j] > matrixResultCopy[(i-1)*rows_columns[1]+j]))
						{
						    matrixResult[i*rows_columns[1]+j] = matrixResultCopy[(i-1)*rows_columns[1]+j];
						    local_flagCambio = 1;
						}
						if((matrixData[(i+1)*rows_columns[1]+j] == matrixData[i*rows_columns[1]+j]) &&
						    (matrixResult[i*rows_columns[1]+j] > matrixResultCopy[(i+1)*rows_columns[1]+j]))
						{
						    matrixResult[i*rows_columns[1]+j] = matrixResultCopy[(i+1)*rows_columns[1]+j];
						    local_flagCambio = 1;
						}
						if((matrixData[i*rows_columns[1]+j-1] == matrixData[i*rows_columns[1]+j]) &&
						    (matrixResult[i*rows_columns[1]+j] > matrixResultCopy[i*rows_columns[1]+j-1]))
						{
						    matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j-1];
						    local_flagCambio = 1;
						}
						if((matrixData[i*rows_columns[1]+j+1] == matrixData[i*rows_columns[1]+j]) &&
						    (matrixResult[i*rows_columns[1]+j] > matrixResultCopy[i*rows_columns[1]+j+1]))
						{
						    matrixResult[i*rows_columns[1]+j] = matrixResultCopy[i*rows_columns[1]+j+1];
						    local_flagCambio = 1;
						}
					}
				}
			}
		} else if (world_size != 1) {
			if (world_rank != world_size - 1) {
				MPI_Wait(&request[2], MPI_STATUS_IGNORE);
				MPI_Start(&request[2]);
			}
			if (world_rank != 0) {
				MPI_Wait(&request[3], MPI_STATUS_IGNORE);
				MPI_Start(&request[3]);
			}
		}

		if (world_size != 1) {
			if(t !=  0){
				MPI_Wait(&request[4], MPI_STATUS_IGNORE);
			}

			flagCambio=0;
			MPI_Iallreduce(&local_flagCambio, &flagCambio, 1, MPI_CHAR, MPI_LOR,
				MPI_COMM_WORLD, &request[4]);

			if (!local_flagCambio){
				MPI_Wait(&request[4], MPI_STATUS_IGNORE);
			}
		}


		#ifdef DEBUG
		printf("\nResultados iter %d: \n", t);
		for(i=0;i<rows_columns[0];i++){
			for(j=0;j<rows_columns[1];j++){
				printf ("%d\t", matrixResult[i*rows_columns[1]+j]);
			}
			printf("\n");
		}
		#endif

	}

	MPI_Type_free(&row_type);

	/* 4.3 Inicio cuenta del numero de bloques */
	local_numBlocks = 0;
	for(i=row_init;i<row_end;i++){
		const int t1 = (i+row_shift*world_rank)*rows_columns[1];
		for(j=1;j<rows_columns[1]-1;j++){
			if(matrixResult[i*rows_columns[1]+j] == t1+j) local_numBlocks++;
		}
	}

	MPI_Reduce(&local_numBlocks, &numBlocks, 1, MPI_INT, MPI_SUM, 0,
       MPI_COMM_WORLD);



	//
	// EL CODIGO A PARALELIZAR TERMINA AQUI
	//
	if ( world_rank == 0 ) {

		/* PUNTO DE FINAL DE MEDIDA DE TIEMPO */
		double t_fin = cp_Wtime();

		/* 5. Comprobación de resultados */
		double t_total = (double)(t_fin - t_ini);

		printf("Result: %d\n", numBlocks);
		printf("Time: %lf\n", t_total);
		#ifdef WRITE
		printf("Resultado: \n");
		for(i=0;i<rows_columns[0];i++){
			for(j=0;j<rows_columns[1];j++){
				printf ("%d\t", matrixResult[i*rows_columns[1]+j]);
			}
			printf("\n");
		}
		#endif

	}

	/* 6. Liberacion de memoria */
	free(matrixData);
	free(matrixResult);
	free(matrixResultCopy);
	free(request);

	MPI_Finalize();
	return 0;
}
