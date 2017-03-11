/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código openmp
*
* @author Sergio García Prado
* @author Adrián Calvo Rojo
* @author Ana Moretón Fernández
* @version v2.0
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "cputils.h"
#include <omp.h>

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

	int k=-1, k_max=-1;
	int *k_indexer=NULL;
	int *temp=NULL;
	char flagCambio =-1;

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
	int i,j,valor;
	fscanf (f, "%d\n", &rows);
	fscanf (f, "%d\n", &columns);
	// Añado dos filas y dos columnas mas para los bordes
	rows=rows+2;
	columns = columns+2;

	/* 2.3 Reservo la memoria necesaria para la matriz de datos */
	matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
	if ( matrixData == NULL ) {
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
	double t_ini = cp_Wtime();

//
// EL CODIGO A PARALELIZAR COMIENZA AQUI
//


	#pragma omp parallel \
	default(none), \
	shared(k_indexer, k_max, matrixData, matrixResult, temp, \
		matrixResultCopy, columns,rows, flagCambio,numBlocks)
	{
		/* 3. Etiquetado inicial */
		#pragma omp single
		{
			matrixResult = (int *)malloc( (rows)*(columns) * sizeof(int) );
			matrixResultCopy = (int *)malloc( (rows)*(columns) * sizeof(int) );
			k_indexer = (int *)malloc( (rows-1)*(columns-1) * sizeof(int) );
			k_max = 0;
			numBlocks=0;

			if ( (matrixResult == NULL)  || (matrixResultCopy == NULL) || (k_indexer == NULL)  ) {
				perror ("Error reservando memoria");
				//return -1;
			}
		}

		#pragma omp for \
		nowait,\
		schedule(dynamic, ((rows-1)*(columns-1))/omp_get_num_threads()), \
		private(i,j,k)
		for(i = 1; i < rows-1; i++){
			for(j = 1; j < columns-1; j++){
				// Si es 0 se trata del fondo y no lo computamos
				if(matrixData[i*(columns)+j]!=0){
					matrixResult[i*(columns)+j]=i*(columns)+j;
					#pragma omp atomic capture
					{
						k = k_max; k_max++;
					}
					k_indexer[k] = i*(columns)+j;
				} else {
					matrixResult[i*(columns)+j]=-1;
				}
			}
			matrixResult[i*(columns)]=-1;
			matrixResult[i*(columns)+columns-1]=-1;
		}

		#pragma omp for \
		nowait,\
		schedule(static), \
		private(j)
		for(j=1;j< columns-1; j++){
			matrixResult[j]=-1;
			matrixResult[(rows-1)*(columns)+j]=-1;
		}

		/* 4.2 Busqueda de los bloques similiares */
		do {
			#pragma omp barrier

			/* 4.2.1 Actualizacion copia */
			#pragma omp single
			{
				flagCambio=0;
				temp = matrixResultCopy;
				matrixResultCopy = matrixResult;
				matrixResult = temp;
			}

			/* 4.2.2 Computo y detecto si ha habido cambios */
			#pragma omp for \
			schedule(dynamic, k_max/omp_get_num_threads()), \
			nowait, \
			reduction(||:flagCambio),\
			private(k)
			for(k=0;k<k_max;k++){
				matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]];
				if((matrixData[k_indexer[k]-columns] == matrixData[k_indexer[k]]) &&
					(matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]-columns]))
				{
					#pragma omp atomic write
					matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]-columns];
					flagCambio = 1;
				}
			}

			#pragma omp for \
			schedule(dynamic, k_max/omp_get_num_threads()), \
			nowait, \
			reduction(||:flagCambio),\
			private(k)
			for(k=0;k<k_max;k++){
				if((matrixData[k_indexer[k]+columns] == matrixData[k_indexer[k]]) &&
					(matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]+columns]))
				{
					#pragma omp atomic write
					matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]+columns];
					flagCambio = 1;
				}
			}

			#pragma omp for \
			schedule(dynamic, k_max/omp_get_num_threads()), \
			nowait, \
			reduction(||:flagCambio),\
			private(k)
			for(k=0;k<k_max;k++){
				if((matrixData[k_indexer[k]-1] == matrixData[k_indexer[k]]) &&
					(matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]-1]))
				{
					#pragma omp atomic write
					matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]-1];
					flagCambio = 1;
				}
			}

			#pragma omp for \
			schedule(dynamic, k_max/omp_get_num_threads()), \
			nowait, \
			reduction(||:flagCambio),\
			private(k)
			for(k=0;k<k_max;k++){
				if((matrixData[k_indexer[k]+1] == matrixData[k_indexer[k]]) &&
					(matrixResult[k_indexer[k]] > matrixResultCopy[k_indexer[k]+1]))
				{
					#pragma omp atomic write
					matrixResult[k_indexer[k]] = matrixResultCopy[k_indexer[k]+1];
					flagCambio = 1;
				}
			}

			#ifdef DEBUG
				#pragma omp for \
				schedule(static), \
				ordered,\
				private(i,j)
				for(i=0;i<rows;i++){
					for(j=0;j<columns;j++){
						printf ("%d\t", matrixResult[i*columns+j]);
					}
					printf("\n");
				}
			#endif
		} while(flagCambio !=0);


		/* 4.3 Inicio cuenta del numero de bloques */
		#pragma omp for \
		schedule(dynamic, k_max/omp_get_num_threads()), \
		private(k),\
		reduction(+:numBlocks)
		for(k=0;k<k_max;k++){
			if(matrixResult[k_indexer[k]] == k_indexer[k]) numBlocks++;
		}
	}

//
// EL CODIGO A PARALELIZAR TERMINA AQUI
//

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

	/* 6. Liberacion de memoria */
	free(matrixData);
	free(matrixResult);
	free(matrixResultCopy);
	free(k_indexer);
}
