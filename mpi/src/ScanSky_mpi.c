/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código secuencial base
*
* @author Ana Moretón Fernández
* @author Eduardo Rodríguez Gutiez
* @version v1.3
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "cputils.h"
#include <mpi.h>


/* Substituir min por el operador */
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
* Funcion secuencial para la busqueda de mi bloque
*/
char computation(int x, int y, int columns, int* matrixData, int *matrixResult,
	int *matrixResultCopy){

	// Inicialmente cojo mi indice
	int result=matrixResultCopy[x*columns+y];
	if( result!= -1){
		//Si es de mi mismo grupo, entonces actualizo
		if(matrixData[(x-1)*columns+y] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[(x-1)*columns+y]);
		}
		if(matrixData[(x+1)*columns+y] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[(x+1)*columns+y]);
		}
		if(matrixData[x*columns+y-1] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[x*columns+y-1]);
		}
		if(matrixData[x*columns+y+1] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[x*columns+y+1]);
		}

		// Si el indice no ha cambiado retorna 0
		if(matrixResult[x*columns+y] == result){ return 0; }
		// Si el indice cambia, actualizo matrix de resultados con el indice adecuado y retorno 1
		else { matrixResult[x*columns+y]=result; return 1;}

	}
	return 0;
}

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

	int row_shift =-1;
	int column_shift =-1;

	int row_init =-1;
	int row_end =-1;

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

		MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

		MPI_Bcast(matrixData, rows*columns, MPI_INT, 0, MPI_COMM_WORLD);

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
	} else {
		MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
		MPI_Bcast(matrixData, rows*columns, MPI_INT, 0, MPI_COMM_WORLD);
	}
	row_shift = (rows-1)/world_size;

	row_init = 1 + row_shift*world_rank;
	row_end = row_shift + row_shift*world_rank+1;

	if(world_rank == world_size-1){
		row_end = rows-1;
	}

	//
	// EL CODIGO A PARALELIZAR COMIENZA AQUI
	//

	/* 3. Etiquetado inicial */
	matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int) );
	matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
	if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
		perror ("Error reservando memoria");
		return -1;
	}
	for(i=row_init-1;i< row_end+1; i++){
		for(j=0;j< columns; j++){
			matrixResult[i*(columns)+j]=-1;
			// Si es 0 se trata del fondo y no lo computamos
			if(matrixData[i*(columns)+j]!=0){
				matrixResult[i*(columns)+j]=i*(columns)+j;
			}
		}
	}


	/* 4. Computacion */
	int t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	char flagCambio=1;
	char local_flagCambio=1;
	MPI_Request *request = (MPI_Request *)malloc( 4 * sizeof(MPI_Request) );

	/* 4.2 Busqueda de los bloques similiares */
	for(t=0; flagCambio !=0; t++){
		flagCambio=0;
		local_flagCambio = 0;

		/* 4.2.1 Actualizacion copia */
		if (world_size > 1) {
			if (world_rank < world_size - 1) {
				MPI_Isend(&matrixResult[(row_end-1)*columns], columns,
					MPI_INT, world_right, 0, MPI_COMM_WORLD, &request[2]);
				MPI_Irecv(&matrixResultCopy[(row_end)*columns], columns,
					MPI_INT, world_right, 0, MPI_COMM_WORLD, &request[0]);
			}
			if (world_rank > 0) {
				MPI_Isend(&matrixResult[(row_init)*columns], columns,
					MPI_INT, world_left, 0, MPI_COMM_WORLD, &request[3]);
				MPI_Irecv(&matrixResultCopy[(row_init-1)*columns], columns,
					MPI_INT, world_left, 0, MPI_COMM_WORLD, &request[1]);
			}

			if (world_rank == world_size -1 ) {
				MPI_Wait(&request[1], MPI_STATUS_IGNORE);
			} else if (world_rank == 0 ) {
				MPI_Wait(&request[0], MPI_STATUS_IGNORE);
			} else {
				MPI_Waitall(2, request, MPI_STATUS_IGNORE);
			}
		}
		temp = matrixResultCopy;
		matrixResultCopy = matrixResult;
		matrixResult = temp;

		/* 4.2.2 Computo y detecto si ha habido cambios */
		for(i=row_init;i<row_end;i++){
			for(j=1;j<columns-1;j++){
				if(matrixResult[i*columns+j] != -1){
					matrixResult[i*columns+j] = matrixResultCopy[i*columns+j];
					if((matrixData[(i-1)*columns+j] == matrixData[i*columns+j]) &&
					    (matrixResult[i*columns+j] > matrixResultCopy[(i-1)*columns+j]))
					{
					    matrixResult[i*columns+j] = matrixResultCopy[(i-1)*columns+j];
					    local_flagCambio = 1;
					}
					if((matrixData[(i+1)*columns+j] == matrixData[i*columns+j]) &&
					    (matrixResult[i*columns+j] > matrixResultCopy[(i+1)*columns+j]))
					{
					    matrixResult[i*columns+j] = matrixResultCopy[(i+1)*columns+j];
					    local_flagCambio = 1;
					}
					if((matrixData[i*columns+j-1] == matrixData[i*columns+j]) &&
					    (matrixResult[i*columns+j] > matrixResultCopy[i*columns+j-1]))
					{
					    matrixResult[i*columns+j] = matrixResultCopy[i*columns+j-1];
					    local_flagCambio = 1;
					}
					if((matrixData[i*columns+j+1] == matrixData[i*columns+j]) &&
					    (matrixResult[i*columns+j] > matrixResultCopy[i*columns+j+1]))
					{
					    matrixResult[i*columns+j] = matrixResultCopy[i*columns+j+1];
					    local_flagCambio = 1;
					}
				}
			}
		}

		MPI_Allreduce(&local_flagCambio, &flagCambio, 1, MPI_CHAR, MPI_LOR,
			MPI_COMM_WORLD);

		#ifdef DEBUG
		printf("\nResultados iter %d: \n", t);
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixResult[i*columns+j]);
			}
			printf("\n");
		}
		#endif

	}

	/* 4.3 Inicio cuenta del numero de bloques */
	local_numBlocks = 0;
	for(i=row_init;i<row_end;i++){
		for(j=1;j<columns-1;j++){
			if(matrixResult[i*columns+j] == i*columns+j) local_numBlocks++;
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
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixResult[i*columns+j]);
			}
			printf("\n");
		}
		#endif

	}

	/* 6. Liberacion de memoria */
	free(matrixData);
	free(matrixResult);
	free(matrixResultCopy);

	MPI_Finalize();
	return 0;
}
