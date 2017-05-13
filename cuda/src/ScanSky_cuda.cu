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



__global__ void kernelFillMatrixResult(int *matrixResult, int *matrixResultCopy,
	int *matrixData, int *rows, int *columns) {

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
	int t=-1;
	char flagCambio=-1;

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

	int rowsBloqShape = 16;
	int columnsBloqShape = 16;

	int rowsGridShape = ceil((float) rows / rowsBloqShape);
	int columnsGridShape = ceil((float) columns / columnsBloqShape);

	dim3 bloqShapeGpu(rowsBloqShape,columnsBloqShape,1);
	dim3 gridShapeGpu(rowsGridShape,columnsGridShape,1);

	cudaMalloc(&rowsDevice, sizeof(rows));
	cudaMalloc(&columnsDevice, sizeof(columns));
	cudaMemcpy(rowsDevice,&rows, sizeof(rows),cudaMemcpyHostToDevice);
	cudaMemcpy(columnsDevice,&columns, sizeof(columns),cudaMemcpyHostToDevice);


	cudaMalloc(&numBlocksDevice, sizeof(numBlocks));
	cudaMalloc(&flagCambioDevice, sizeof(flagCambio));

	cudaMalloc( (void**) &matrixDataDevice, sizeof(int) * rows * columns);
	cudaMemcpy(matrixDataDevice,matrixData, sizeof(int) * rows * columns,cudaMemcpyHostToDevice);


	/* 3. Etiquetado inicial */
	cudaMalloc( (void**) &matrixResultDevice, sizeof(int) * sizeof(int) * rows * columns);
	cudaMalloc( (void**) &matrixResultCopyDevice, sizeof(int) * sizeof(int) * rows * columns);

	kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
		matrixResultCopyDevice, matrixDataDevice, rowsDevice, columnsDevice);

	/* 4. Computacion */
	t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	flagCambio=1;

	for(t=0; flagCambio !=0; t++){

		cudaMemset(flagCambioDevice,0,sizeof(flagCambio));

		temp = matrixResultDevice;
		matrixResultDevice = matrixResultCopyDevice;
		matrixResultCopyDevice = temp;

		kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
			matrixResultCopyDevice, flagCambioDevice, matrixDataDevice,
			rowsDevice, columnsDevice);

		cudaMemcpy(&flagCambio,flagCambioDevice, sizeof(flagCambio),cudaMemcpyDeviceToHost);
	}


	/* 4.3 Inicio cuenta del numero de bloques */
	cudaMemset(numBlocksDevice,0,sizeof(int));

	kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>(matrixResultDevice,
		numBlocksDevice, rowsDevice, columnsDevice);

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
	cudaFree(flagCambioDevice);
	cudaFree(matrixDataDevice);
	cudaFree(matrixResultDevice);
	cudaFree(matrixResultCopyDevice);

	/*Liberamos los hilos del DEVICE*/
	cudaDeviceReset();
	return 0;
}
