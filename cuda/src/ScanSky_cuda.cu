/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código CUDA
*
* @author Ana Moretón Fernández,
* @author Arturo Gonzalez-Escribano
* @author Sergio García Prado (@garciparedes)
* @author Adrian Calvo Rojo
* @version v3.0
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "cputils.h"


/* Substituir min por el operador */
#define min(x,y)    ((x) < (y)? (x) : (y))


__global__ void kernelFillMatrixResult(int *matrixResult, int *matrixResultCopy,
	int *matrixData, int *rows, int *columns) {
	/*
	int index = threadIdx.x +
				threadIdx.y * blockDim.x +
        		blockIdx.x * blockDim.x * blockDim.y +
        		blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
	*/

 	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > -1 && i<rows[0] &&
		j > -1 && j<columns[0]){
		// Si es 0 se trata del fondo y no lo computamos
		if(matrixData[i*(columns[0])+j] !=0){
			matrixResult[i*(columns[0])+j]=i*(columns[0])+j;
			matrixResultCopy[i*(columns[0])+j]=i*(columns[0])+j;
		} else {
			matrixResult[i*(columns[0])+j]=-1;
			matrixResultCopy[i*(columns[0])+j]=-1;
		}
	}
}

__global__ void kernelComputationLoop(int *matrixResult,int *matrixResultCopy,
	int *flagCambio, int *matrixData, int *rows, int *columns) {
	/*
	int index = threadIdx.x +
				threadIdx.y * blockDim.x +
				blockIdx.x * blockDim.x * blockDim.y +
				blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
	*/


 	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	/* 4.2.2 Computo y detecto si ha habido cambios */
	if(i > 0 && i<rows[0]-1 &&
		j > 0 && j<columns[0]-1){
		if(matrixResult[i*columns[0]+j] != -1){
			matrixResult[i*columns[0]+j] = matrixResultCopy[i*columns[0]+j];
			if((matrixData[(i-1)*columns[0]+j] == matrixData[i*columns[0]+j]) &&
				(matrixResult[i*columns[0]+j] > matrixResultCopy[(i-1)*columns[0]+j]))
			{
				matrixResult[i*columns[0]+j] = matrixResultCopy[(i-1)*columns[0]+j];
				atomicOr(&flagCambio[0], (int)1);
			}
			if((matrixData[(i+1)*columns[0]+j] == matrixData[i*columns[0]+j]) &&
				(matrixResult[i*columns[0]+j] > matrixResultCopy[(i+1)*columns[0]+j]))
			{
				matrixResult[i*columns[0]+j] = matrixResultCopy[(i+1)*columns[0]+j];
				atomicOr(&flagCambio[0], (int)1);
			}
			if((matrixData[i*columns[0]+j-1] == matrixData[i*columns[0]+j]) &&
				(matrixResult[i*columns[0]+j] > matrixResultCopy[i*columns[0]+j-1]))
			{
				matrixResult[i*columns[0]+j] = matrixResultCopy[i*columns[0]+j-1];
				atomicOr(&flagCambio[0], (int)1);
			}
			if((matrixData[i*columns[0]+j+1] == matrixData[i*columns[0]+j]) &&
				(matrixResult[i*columns[0]+j] > matrixResultCopy[i*columns[0]+j+1]))
			{
				matrixResult[i*columns[0]+j] = matrixResultCopy[i*columns[0]+j+1];
				atomicOr(&flagCambio[0], (int)1);
			}
		}
	}
}

__global__ void kernelCountFigures(int *matrixResult, int *count,
	int *rows, int *columns) {
	/*
	int index = threadIdx.x +
				threadIdx.y * blockDim.x +
        		blockIdx.x * blockDim.x * blockDim.y +
        		blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
	*/

 	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > 0 && i<rows[0]-1 &&
		j > 0 && j<columns[0]-1){

		if (matrixResult[i*columns[0]+j] == i*columns[0]+j) {
			atomicAdd(&count[0],(int) 1);
		}
	}
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

	int *rowsDevice;
	int *columnsDevice;
	int *matrixDataDevice;
	int *matrixResultDevice;
	int *matrixResultCopyDevice;
	int *temp;
	int *numBlocksDevice;
	int *flagCambioDevice;


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
	int i,j;
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

	cudaSetDevice(0);
	cudaDeviceSynchronize();

	/* PUNTO DE INICIO MEDIDA DE TIEMPO */
	double t_ini = cp_Wtime();

//
// EL CODIGO A PARALELIZAR COMIENZA AQUI
//

	int rowsBloqShape = 32;
	int columnsBloqShape = 32;

	int rowsGridShape = 128;
	int columnsGridShape = 128;

	dim3 bloqShapeGpu(rowsBloqShape,columnsBloqShape,1);
	dim3 gridShapeGpu(rowsGridShape,columnsGridShape,1);

	cudaMalloc(&rowsDevice, sizeof(int));
	cudaMalloc(&columnsDevice, sizeof(int));

	cudaMalloc(&numBlocksDevice, sizeof(int));
	cudaMalloc(&flagCambioDevice, sizeof(int));

	cudaMalloc( (void**) &matrixDataDevice, sizeof(int) * (int)((rows)*(columns)));

	cudaMemcpy(rowsDevice,&rows, sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(columnsDevice,&columns, sizeof(int),cudaMemcpyHostToDevice);

	cudaMemcpy(matrixDataDevice,matrixData, sizeof(int) * (int)((rows)*(columns)),cudaMemcpyHostToDevice);


	/* 3. Etiquetado inicial */
	matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int) );
	matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
	cudaMalloc( (void**) &matrixResultDevice, sizeof(int) * (int)((rows)*(columns)));
	cudaMalloc( (void**) &matrixResultCopyDevice, sizeof(int) * (int)((rows)*(columns)));

	if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
 		perror ("Error reservando memoria");
	   	return -1;
	}

	kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
		matrixResultCopyDevice, matrixDataDevice, rowsDevice, columnsDevice);

	//cudaMemcpy(matrixResult,matrixResultDevice, sizeof(int) * (int)((rows)*(columns)),cudaMemcpyDeviceToHost);

	/* 4. Computacion */
	int t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	int flagCambio=1;

	cudaMemcpy(flagCambioDevice,&flagCambio, sizeof(int),cudaMemcpyHostToDevice);

	/*
	kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
		matrixResultCopyDevice, flagCambioDevice, matrixDataDevice,
		rowsDevice, columnsDevice);
	*/
	for(t=0; flagCambio !=0; t++){
		/*
		flagCambio=0;

		temp = matrixResult;
		matrixResult = matrixResultCopy;
		matrixResultCopy = temp;

		for(i=1;i<rows-1;i++){
			for(j=1;j<columns-1;j++){
				if(matrixResult[i*columns+j] != -1){
					matrixResult[i*columns+j] = matrixResultCopy[i*columns+j];
					if((matrixData[(i-1)*columns+j] == matrixData[i*columns+j]) &&
						(matrixResult[i*columns+j] > matrixResultCopy[(i-1)*columns+j]))
					{
						matrixResult[i*columns+j] = matrixResultCopy[(i-1)*columns+j];
						flagCambio = 1;
					}
					if((matrixData[(i+1)*columns+j] == matrixData[i*columns+j]) &&
						(matrixResult[i*columns+j] > matrixResultCopy[(i+1)*columns+j]))
					{
						matrixResult[i*columns+j] = matrixResultCopy[(i+1)*columns+j];
						flagCambio = 1;
					}
					if((matrixData[i*columns+j-1] == matrixData[i*columns+j]) &&
						(matrixResult[i*columns+j] > matrixResultCopy[i*columns+j-1]))
					{
						matrixResult[i*columns+j] = matrixResultCopy[i*columns+j-1];
						flagCambio = 1;
					}
					if((matrixData[i*columns+j+1] == matrixData[i*columns+j]) &&
						(matrixResult[i*columns+j] > matrixResultCopy[i*columns+j+1]))
					{
						matrixResult[i*columns+j] = matrixResultCopy[i*columns+j+1];
						flagCambio = 1;
					}
				}
			}
		}
		*/

		cudaMemset(flagCambioDevice,0,sizeof(int));

		temp = matrixResultDevice;
		matrixResultDevice = matrixResultCopyDevice;
		matrixResultCopyDevice = temp;

		kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
			matrixResultCopyDevice, flagCambioDevice, matrixDataDevice,
			rowsDevice, columnsDevice);


		cudaMemcpy(&flagCambio,flagCambioDevice, sizeof(int),cudaMemcpyDeviceToHost);

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

	//cudaMemcpy(matrixResultDevice,matrixResult, sizeof(int) * (int)((rows)*(columns)),cudaMemcpyHostToDevice);


	/* 4.3 Inicio cuenta del numero de bloques */
	cudaMemset(numBlocksDevice,0,sizeof(int));

	kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice, numBlocksDevice, rowsDevice, columnsDevice);

	cudaMemcpy(&numBlocks,numBlocksDevice, sizeof(int),cudaMemcpyDeviceToHost);

//
// EL CODIGO A PARALELIZAR TERMINA AQUI
//

	/* PUNTO DE FINAL DE MEDIDA DE TIEMPO */
	cudaDeviceSynchronize();
 	double t_fin = cp_Wtime();


	/* 5. Comprobación de resultados */
  	double t_total = (double)(t_fin - t_ini);

	printf("Result: %d:%d\n", numBlocks, t);
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

	/*Liberamos memoria del DEVICE*/
	cudaFree(rowsDevice);
	cudaFree(columnsDevice);
	cudaFree(numBlocksDevice);
	cudaFree(matrixResultDevice);
	cudaFree(matrixResultCopyDevice);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
	return 0;
}
