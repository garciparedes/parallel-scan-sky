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
#define nStreams 4

/*
*
* CUDA MEMCHECK
* code from: http://www.orangeowlsolutions.com/archives/613
*/
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
    }
}


__device__ __constant__ int rowsDevice;
__device__ __constant__ int columnsDevice;
__device__ __constant__ char* matrixDataPointer;
__device__ int numBlocksDevice;

__global__ void kernelFillMatrixResult(int *matrixResult) {

    const int ij = (blockIdx.y * blockDim.y + threadIdx.y)*columnsDevice +
                        blockIdx.x * blockDim.x + threadIdx.x;

	if(ij > -1 && ij<rowsDevice*columnsDevice){
		if(matrixDataPointer[ij] !=0){
            matrixResult[ij]=ij;
		} else {
            matrixResult[ij]=-1;
		}
	}
}

__global__ void kernelComputationLoop(int *matrixResult,int *matrixResultCopy, char *flagCambioDevice) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y+1;
	const int j = blockIdx.x * blockDim.x + threadIdx.x+1;

	/* 4.2.2 Computo y detecto si ha habido cambios */
	if(i > 0 && i<rowsDevice-1 &&
		j > 0 && j<columnsDevice-1){

		if(matrixResult[i*columnsDevice+j] != -1){

			matrixResult[i*columnsDevice+j] = matrixResultCopy[i*columnsDevice+j];
			if((matrixDataPointer[(i-1)*columnsDevice+j] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResult[i*columnsDevice+j] > matrixResultCopy[(i-1)*columnsDevice+j]))
			{
				matrixResult[i*columnsDevice+j] = matrixResultCopy[(i-1)*columnsDevice+j];
				flagCambioDevice[0] = 1;
			}
			if((matrixDataPointer[(i+1)*columnsDevice+j] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResult[i*columnsDevice+j] > matrixResultCopy[(i+1)*columnsDevice+j]))
			{
				matrixResult[i*columnsDevice+j] = matrixResultCopy[(i+1)*columnsDevice+j];
				flagCambioDevice[0] = 1;
			}
			if((matrixDataPointer[i*columnsDevice+j-1] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResult[i*columnsDevice+j] > matrixResultCopy[i*columnsDevice+j-1]))
			{
				matrixResult[i*columnsDevice+j] = matrixResultCopy[i*columnsDevice+j-1];
				flagCambioDevice[0] = 1;
			}
			if((matrixDataPointer[i*columnsDevice+j+1] == matrixDataPointer[i*columnsDevice+j]) &&
				(matrixResult[i*columnsDevice+j] > matrixResultCopy[i*columnsDevice+j+1]))
			{
				matrixResult[i*columnsDevice+j] = matrixResultCopy[i*columnsDevice+j+1];
				flagCambioDevice[0] = 1;
			}
		}
	}
}

__global__ void kernelCountFigures(int *matrixResult) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y+1;
 	const int j = blockIdx.x * blockDim.x + threadIdx.x+1;

	if(i > 0 && i<rowsDevice-1 &&
		j > 0 && j<columnsDevice-1 &&
			matrixResult[i*columnsDevice+j] == i*columnsDevice+j) {
				atomicAdd(&numBlocksDevice, 1);
	}
}

/**
* Funcion principal
*/
int main (int argc, char* argv[])
{
    const char zero = 0;

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
	int *matrixResult1Device;
    int *matrixResult2Device;
    int *matrixResult3Device;
    int *matrixResult4Device;
    char *flagCambioDevice1;
    char *flagCambioDevice2;
    char *flagCambioDevice3;
    char *flagCambioDevice4;

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

	gpuErrorCheck(cudaSetDevice(0));
	gpuErrorCheck(cudaDeviceSynchronize());

	/* PUNTO DE INICIO MEDIDA DE TIEMPO */
	double t_ini = cp_Wtime();

//
// EL CODIGO A PARALELIZAR COMIENZA AQUI
//
  cudaStream_t stream[4];
  gpuErrorCheck( cudaStreamCreate(&stream[0]) );
  gpuErrorCheck( cudaStreamCreate(&stream[1]) );
  gpuErrorCheck( cudaStreamCreate(&stream[2]) );
  gpuErrorCheck( cudaStreamCreate(&stream[3]) );

	const dim3 bloqShapeGpu(columnsBloqShape,rowsBloqShape);
	const dim3 gridShapeGpu(
		ceil((float) columns / columnsBloqShape),
		ceil((float) rows / rowsBloqShape)
	);

	size_t pitch,pitch3;

  const dim3 gridShapeGpuMin(
      ceil((float) (columns-1) / columnsBloqShape),
      ceil((float) (rows-1) / rowsBloqShape)
  );

  gpuErrorCheck(cudaMalloc(&flagCambioDevice1, sizeof(char)));
  gpuErrorCheck(cudaMalloc(&flagCambioDevice2, sizeof(char)));
  gpuErrorCheck(cudaMalloc(&flagCambioDevice3, sizeof(char)));
  gpuErrorCheck(cudaMalloc(&flagCambioDevice4, sizeof(char)));
  gpuErrorCheck(cudaMallocPitch(&matrixResult1Device, &pitch, rows*sizeof(int), columns));
  gpuErrorCheck(cudaMallocPitch(&matrixResult2Device, &pitch, rows*sizeof(int), columns));
  gpuErrorCheck(cudaMallocPitch(&matrixResult3Device, &pitch, rows*sizeof(int), columns));
  gpuErrorCheck(cudaMallocPitch(&matrixResult4Device, &pitch, rows*sizeof(int), columns));
	gpuErrorCheck(cudaMallocPitch(&matrixDataDevice, &pitch3, rows*sizeof(char), columns));
	//gpuErrorCheck(cudaMalloc(&matrixDataDevice, sizeof(char) * rows * columns));

	gpuErrorCheck(cudaMemcpyToSymbolAsync(rowsDevice,&rows, sizeof(int),0,cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpyToSymbolAsync(columnsDevice,&columns, sizeof(int),0,cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpyToSymbolAsync(matrixDataPointer,&matrixDataDevice, sizeof(char *)));

	//gpuErrorCheck(cudaMallocHost(&matrixDataChar, rows*(columns) * sizeof(char)));
	matrixDataChar= (char *)malloc( rows*(columns) * sizeof(char) );
	for(i = 0; i < rows * columns; i++){
		matrixDataChar[i] = matrixData[i];
	}
	/*
	gpuErrorCheck(cudaMemcpy2D(
		matrixDataDevice,
		pitch3,
		matrixDataChar,
		rows*sizeof(char),
		rows*sizeof(char),
		columns,
		cudaMemcpyHostToDevice
	));
	*/
	gpuErrorCheck(cudaMemcpyAsync(matrixDataDevice,matrixDataChar,
        sizeof(char) * rows * columns,cudaMemcpyHostToDevice));


	/* 3. Etiquetado inicial */


	kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(matrixResult4Device);
	gpuErrorCheck(cudaPeekAtLastError());

	/* 4. Computacion */
	t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	flagCambio=1;

  gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice1,&zero, sizeof(char),
      cudaMemcpyHostToDevice,stream[0]));
  kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[0]>>>(
      matrixResult1Device,matrixResult4Device,flagCambioDevice1);

  gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice2,&zero, sizeof(char),
      cudaMemcpyHostToDevice,stream[0]));
  kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[0]>>>(
      matrixResult2Device,matrixResult1Device,flagCambioDevice2);

  gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice3,&zero, sizeof(char),
      cudaMemcpyHostToDevice,stream[0]));
  kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[0]>>>(
      matrixResult3Device,matrixResult2Device,flagCambioDevice3);

  gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice4,&zero, sizeof(char),
      cudaMemcpyHostToDevice,stream[0]));
  kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[0]>>>(
      matrixResult4Device,matrixResult3Device,flagCambioDevice4);

  for(t=0; flagCambio != 0; t++){
  flagCambio = 0;
      if(t % nStreams == 0){
          gpuErrorCheck(cudaMemcpyAsync(&flagCambio,flagCambioDevice1, sizeof(char),
              cudaMemcpyDeviceToHost,stream[0]));
          gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice1,&zero, sizeof(char),
              cudaMemcpyHostToDevice,stream[0]));
          kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[0]>>>(
              matrixResult1Device,matrixResult4Device,flagCambioDevice1);
      } else if(t % nStreams == 1){
          gpuErrorCheck(cudaMemcpyAsync(&flagCambio,flagCambioDevice2, sizeof(char),
              cudaMemcpyDeviceToHost,stream[1]));
          gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice2,&zero, sizeof(char),
              cudaMemcpyHostToDevice,stream[1]));
          kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[1]>>>(
              matrixResult2Device,matrixResult1Device,flagCambioDevice2);
      } else if(t % nStreams == 2){
          gpuErrorCheck(cudaMemcpyAsync(&flagCambio,flagCambioDevice3, sizeof(char),
              cudaMemcpyDeviceToHost,stream[2]));
          gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice3,&zero, sizeof(char),
              cudaMemcpyHostToDevice,stream[2]));
          kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[2]>>>(
              matrixResult3Device,matrixResult2Device,flagCambioDevice3);
      } else {
          gpuErrorCheck(cudaMemcpyAsync(&flagCambio,flagCambioDevice4, sizeof(char),
              cudaMemcpyDeviceToHost,stream[3]));
          gpuErrorCheck(cudaMemcpyAsync(flagCambioDevice4,&zero, sizeof(char),
              cudaMemcpyHostToDevice,stream[3]));
          kernelComputationLoop<<<gridShapeGpuMin, bloqShapeGpu,0,stream[3]>>>(
              matrixResult4Device,matrixResult3Device,flagCambioDevice4);
      }
	}


	/* 4.3 Inicio cuenta del numero de bloques */
	numBlocks = 0;
	gpuErrorCheck(cudaMemcpyToSymbolAsync(numBlocksDevice,&numBlocks,
        sizeof(int),0,cudaMemcpyHostToDevice));
    if(t % nStreams == 0){
        kernelCountFigures<<<gridShapeGpuMin, bloqShapeGpu>>>(matrixResult1Device);
    } else if(t % nStreams == 1){
        kernelCountFigures<<<gridShapeGpuMin, bloqShapeGpu>>>(matrixResult2Device);
    } else if(t % nStreams == 2){
        kernelCountFigures<<<gridShapeGpuMin, bloqShapeGpu>>>(matrixResult3Device);
    } else {
        kernelCountFigures<<<gridShapeGpuMin, bloqShapeGpu>>>(matrixResult4Device);
    }

	gpuErrorCheck(cudaPeekAtLastError());

	gpuErrorCheck(cudaMemcpyFromSymbolAsync(&numBlocks, numBlocksDevice,
        sizeof(int), 0, cudaMemcpyDeviceToHost));

//
// EL CODIGO A PARALELIZAR TERMINA AQUI
//

	/* PUNTO DE FINAL DE MEDIDA DE TIEMPO */
	gpuErrorCheck(cudaDeviceSynchronize());
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
	gpuErrorCheck(cudaFree(matrixDataPointer));
	gpuErrorCheck(cudaFree(matrixDataDevice));
	gpuErrorCheck(cudaFree(matrixResult1Device));
    gpuErrorCheck(cudaFree(matrixResult2Device));
    gpuErrorCheck(cudaFree(matrixResult3Device));
    gpuErrorCheck(cudaFree(matrixResult4Device));
    gpuErrorCheck( cudaStreamDestroy(stream[0]) );
    gpuErrorCheck( cudaStreamDestroy(stream[1]) );
    gpuErrorCheck( cudaStreamDestroy(stream[2]) );
    gpuErrorCheck( cudaStreamDestroy(stream[3]) );

	/*Liberamos los hilos del DEVICE*/
	gpuErrorCheck(cudaDeviceReset());
	return 0;
}
