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

__device__ __constant__ int rowsDevice;
__device__ __constant__ int columnsDevice;
__device__ __constant__ char* matrixDataPointer;
__device__ int numBlocksDevice;
__device__ char flagCambioDevice;
__device__ int* matrixResultPointer;
__device__ int* matrixResultCopyPointer;
__device__ int* temPointer;

__global__ void kernelFillMatrixResult() {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > -1 && i<rowsDevice &&
		j > -1 && j<columnsDevice){
		if(matrixDataPointer[i*(columnsDevice)+j] !=0){
			matrixResultPointer[i*(columnsDevice)+j]=i*(columnsDevice)+j;
			matrixResultCopyPointer[i*(columnsDevice)+j]=i*(columnsDevice)+j;
		} else {
			matrixResultPointer[i*(columnsDevice)+j]=-1;
			matrixResultCopyPointer[i*(columnsDevice)+j]=-1;
		}
	}
}

__global__ void kernelComputationLoop() {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i == 0 && j == 0 ){
		temPointer = matrixResultPointer;
		matrixResultPointer = matrixResultCopyPointer;
		matrixResultCopyPointer = temPointer;
	}
	//__syncthreads();

	/* 4.2.2 Computo y detecto si ha habido cambios */
	if(i > 0 && i<rowsDevice-1 &&
		j > 0 && j<columnsDevice-1){

		if(matrixResultPointer[i*columnsDevice+j] != -1){

			matrixResultPointer[i*columnsDevice+j] = matrixResultCopyPointer[i*columnsDevice+j];
			if((matrixDataPointer[(i-1)*columnsDevice+j] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResultPointer[i*columnsDevice+j] > matrixResultCopyPointer[(i-1)*columnsDevice+j]))
			{
				matrixResultPointer[i*columnsDevice+j] = matrixResultCopyPointer[(i-1)*columnsDevice+j];
				flagCambioDevice = 1;
			}
			if((matrixDataPointer[(i+1)*columnsDevice+j] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResultPointer[i*columnsDevice+j] > matrixResultCopyPointer[(i+1)*columnsDevice+j]))
			{
				matrixResultPointer[i*columnsDevice+j] = matrixResultCopyPointer[(i+1)*columnsDevice+j];
				flagCambioDevice = 1;
			}
			if((matrixDataPointer[i*columnsDevice+j-1] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResultPointer[i*columnsDevice+j] > matrixResultCopyPointer[i*columnsDevice+j-1]))
			{
				matrixResultPointer[i*columnsDevice+j] = matrixResultCopyPointer[i*columnsDevice+j-1];
				flagCambioDevice = 1;
			}
			if((matrixDataPointer[i*columnsDevice+j+1] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResultPointer[i*columnsDevice+j] > matrixResultCopyPointer[i*columnsDevice+j+1]))
			{
				matrixResultPointer[i*columnsDevice+j] = matrixResultCopyPointer[i*columnsDevice+j+1];
				flagCambioDevice = 1;
			}
		}
	}
}

__global__ void kernelCountFigures() {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
 	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 && i<rowsDevice-1 &&
		j > 0 && j<columnsDevice-1 &&
			matrixResultPointer[i*columnsDevice+j] == i*columnsDevice+j) {
				atomicAdd(&numBlocksDevice, 1);
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


	cudaMalloc( (void**) &matrixResultDevice, sizeof(int) * sizeof(int) * rows * columns);
	cudaMalloc( (void**) &matrixResultCopyDevice, sizeof(int) * sizeof(int) * rows * columns);
	cudaMalloc( (void**) &matrixDataDevice, sizeof(char) * rows * columns);

	cudaMemcpyToSymbolAsync(rowsDevice,&rows, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(columnsDevice,&columns, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(matrixDataPointer,&matrixDataDevice, sizeof(char *));
	cudaMemcpyToSymbolAsync(matrixResultPointer,&matrixResultDevice, sizeof(int *));
	cudaMemcpyToSymbolAsync(matrixResultCopyPointer,&matrixResultCopyDevice, sizeof(int *));

	matrixDataChar = (char *)malloc(rows*(columns) * sizeof(char) );
	for(i = 0; i < rows * columns; i++){
		matrixDataChar[i] = matrixData[i];
	}

	cudaMemcpyAsync(matrixDataDevice,matrixDataChar, sizeof(char) * rows * columns,cudaMemcpyHostToDevice);


	/* 3. Etiquetado inicial */


	kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>();

	/* 4. Computacion */
	t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	flagCambio=1;

	for(t=0; flagCambio != 0; t++){

		flagCambio = 0;
		cudaMemcpyToSymbolAsync(flagCambioDevice,&flagCambio, sizeof(char),0,cudaMemcpyHostToDevice);

		//temp = matrixResultDevice;
		//matrixResultDevice = matrixResultCopyDevice;
		//matrixResultCopyDevice = temp;

		//cudaMemcpyToSymbolAsync(matrixResultPointer,&matrixResultCopyDevice, sizeof(int *));
		//cudaMemcpyToSymbolAsync(matrixResultCopyPointer,&matrixResultDevice, sizeof(int *));

		kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu>>>();
		cudaMemcpyFromSymbol(&flagCambio, flagCambioDevice, sizeof(char), 0, cudaMemcpyDeviceToHost);
	}


	/* 4.3 Inicio cuenta del numero de bloques */
	numBlocks = 0;
	cudaMemcpyToSymbolAsync(numBlocksDevice,&numBlocks, sizeof(int),0,cudaMemcpyHostToDevice);

	kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>();

	cudaMemcpyFromSymbolAsync(&numBlocks, numBlocksDevice, sizeof(int), 0, cudaMemcpyDeviceToHost);

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
	cudaFree(matrixDataPointer);
	cudaFree(matrixDataDevice);
	cudaFree(matrixResultDevice);
	cudaFree(matrixResultCopyDevice);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
	return 0;
}
