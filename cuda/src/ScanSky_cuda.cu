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


#define rowsBloqShape 8
#define columnsBloqShape 16

__device__ __constant__ int rowsDevice[1];
__device__ __constant__ int columnsDevice[1];
__device__ __constant__ char* matrixDataPointer;

__global__ void kernelFillMatrixResult(int *matrixResult, int *matrixResultCopy) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > -1 && i<rowsDevice[0] &&
		j > -1 && j<columnsDevice[0]){
		if(matrixDataPointer[i*(columnsDevice[0])+j] !=0){
			matrixResult[i*(columnsDevice[0])+j]=i*(columnsDevice[0])+j;
			matrixResultCopy[i*(columnsDevice[0])+j]=i*(columnsDevice[0])+j;
		} else {
			matrixResult[i*(columnsDevice[0])+j]=-1;
			matrixResultCopy[i*(columnsDevice[0])+j]=-1;
		}
	}
}

__global__ void kernelComputationLoop(int *matrixResult,int *matrixResultCopy,
	char *flagCambio) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	/* 4.2.2 Computo y detecto si ha habido cambios */
	if(i > 0 && i<rowsDevice[0]-1 &&
		j > 0 && j<columnsDevice[0]-1){

		if(matrixResult[i*columnsDevice[0]+j] != -1){

			matrixResult[i*columnsDevice[0]+j] = matrixResultCopy[i*columnsDevice[0]+j];
			if((matrixDataPointer[(i-1)*columnsDevice[0]+j] == matrixDataPointer[i*columnsDevice[0]+j]) &&
				(matrixResult[i*columnsDevice[0]+j] > matrixResultCopy[(i-1)*columnsDevice[0]+j]))
			{
				matrixResult[i*columnsDevice[0]+j] = matrixResultCopy[(i-1)*columnsDevice[0]+j];
				flagCambio[0] = 1;
			}
			if((matrixDataPointer[(i+1)*columnsDevice[0]+j] == matrixDataPointer[i*columnsDevice[0]+j]) &&
				(matrixResult[i*columnsDevice[0]+j] > matrixResultCopy[(i+1)*columnsDevice[0]+j]))
			{
				matrixResult[i*columnsDevice[0]+j] = matrixResultCopy[(i+1)*columnsDevice[0]+j];
				flagCambio[0] = 1;
			}
			if((matrixDataPointer[i*columnsDevice[0]+j-1] == matrixDataPointer[i*columnsDevice[0]+j]) &&
				(matrixResult[i*columnsDevice[0]+j] > matrixResultCopy[i*columnsDevice[0]+j-1]))
			{
				matrixResult[i*columnsDevice[0]+j] = matrixResultCopy[i*columnsDevice[0]+j-1];
				flagCambio[0] = 1;
			}
			if((matrixDataPointer[i*columnsDevice[0]+j+1] == matrixDataPointer[i*columnsDevice[0]+j]) &&
				(matrixResult[i*columnsDevice[0]+j] > matrixResultCopy[i*columnsDevice[0]+j+1]))
			{
				matrixResult[i*columnsDevice[0]+j] = matrixResultCopy[i*columnsDevice[0]+j+1];
				flagCambio[0] = 1;
			}
		}
	}
}

__global__ void kernelCountFigures(int *matrixResult, int *count) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
 	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 && i<rowsDevice[0]-1 &&
		j > 0 && j<columnsDevice[0]-1 &&
			matrixResult[i*columnsDevice[0]+j] == i*columnsDevice[0]+j) {
				atomicAdd(&count[0],(int) 1);
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
	char *matrixDataChar=NULL;
	int *matrixResult=NULL;
	int *matrixResultCopy=NULL;
	int numBlocks=-1;
	int t=-1;
	char flagCambio=-1;

	char *matrixDataDevice;
	int *matrixResultDevice;
	int *matrixResultCopyDevice;
	int *temp;
	int *numBlocksDevice;
	char *flagCambioDevice;

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


	const dim3 bloqShapeGpu(columnsBloqShape,rowsBloqShape);
	const dim3 gridShapeGpu(
		ceil((float) columns / columnsBloqShape),
		ceil((float) rows / rowsBloqShape)
	);

	cudaMemcpyToSymbolAsync(rowsDevice,&rows, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(columnsDevice,&columns, sizeof(int),0,cudaMemcpyHostToDevice);


	cudaMalloc(&numBlocksDevice, sizeof(int));
	cudaMalloc(&flagCambioDevice, sizeof(int));

	matrixDataChar = (char *)malloc(rows*(columns) * sizeof(char) );
	for(i = 0; i < rows * columns; i++){
		matrixDataChar[i] = matrixData[i];
	}

	cudaMalloc( (void**) &matrixDataDevice, sizeof(char) * rows * columns);


	cudaMemcpyAsync(matrixDataDevice,matrixDataChar, sizeof(char) * rows * columns,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(matrixDataPointer,&matrixDataDevice, sizeof(char *));


	/* 3. Etiquetado inicial */
	cudaMalloc( (void**) &matrixResultDevice, sizeof(int) * sizeof(int) * rows * columns);
	cudaMalloc( (void**) &matrixResultCopyDevice, sizeof(int) * sizeof(int) * rows * columns);

	kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
		matrixResultCopyDevice);

	/* 4. Computacion */
	t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	flagCambio=1;

	for(t=0; flagCambio !=0; t++){

		cudaMemsetAsync(flagCambioDevice,0,sizeof(char));

		temp = matrixResultDevice;
		matrixResultDevice = matrixResultCopyDevice;
		matrixResultCopyDevice = temp;

		kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
			matrixResultCopyDevice, flagCambioDevice);

		cudaMemcpy(&flagCambio,flagCambioDevice, sizeof(char),cudaMemcpyDeviceToHost);
	}


	/* 4.3 Inicio cuenta del numero de bloques */
	cudaMemsetAsync(numBlocksDevice,0,sizeof(int));

	kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
		numBlocksDevice);

	cudaMemcpyAsync(&numBlocks,numBlocksDevice, sizeof(int),cudaMemcpyDeviceToHost);

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
	cudaFree(flagCambioDevice);
	cudaFree(matrixDataPointer);
	cudaFree(matrixDataDevice);
	cudaFree(matrixResultDevice);
	cudaFree(matrixResultCopyDevice);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
	return 0;
}
