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


__device__ __constant__ int rows_d;
__device__ __constant__ int columns_d;
__device__ __constant__ char* matrixData_d;
__device__ int numBlocks_d;

__global__ void kernelFillMatrixResult(int *matrixResult) {

    const int ij = (blockIdx.y * blockDim.y + threadIdx.y)*columns_d +
                        blockIdx.x * blockDim.x + threadIdx.x;

	if(ij > -1 && ij<rows_d*columns_d){
		if(matrixData_d[ij] !=0){
            matrixResult[ij]=ij;
		} else {
            matrixResult[ij]=-1;
		}
	}
}

__global__ void kernelComputationLoop(int *matrixResult,int *matrixResultCopy,
    char *flagCambio_d) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	/* 4.2.2 Computo y detecto si ha habido cambios */
	if(i > 0 && i<rows_d-1 &&
		j > 0 && j<columns_d-1){

		if(matrixResult[i*columns_d+j] != -1){

			matrixResult[i*columns_d+j] = matrixResultCopy[i*columns_d+j];
			if((matrixData_d[(i-1)*columns_d+j] == matrixData_d[i*columns_d+j]) &&
				(matrixResult[i*columns_d+j] > matrixResultCopy[(i-1)*columns_d+j]))
			{
				matrixResult[i*columns_d+j] = matrixResultCopy[(i-1)*columns_d+j];
				*flagCambio_d = 1;
			}
			if((matrixData_d[(i+1)*columns_d+j] == matrixData_d[i*columns_d+j]) &&
				(matrixResult[i*columns_d+j] > matrixResultCopy[(i+1)*columns_d+j]))
			{
				matrixResult[i*columns_d+j] = matrixResultCopy[(i+1)*columns_d+j];
				*flagCambio_d = 1;
			}
			if((matrixData_d[i*columns_d+j-1] == matrixData_d[i*columns_d+j]) &&
				(matrixResult[i*columns_d+j] > matrixResultCopy[i*columns_d+j-1]))
			{
				matrixResult[i*columns_d+j] = matrixResultCopy[i*columns_d+j-1];
				*flagCambio_d = 1;
			}
			if((matrixData_d[i*columns_d+j+1] == matrixData_d[i*columns_d+j]) &&
				(matrixResult[i*columns_d+j] > matrixResultCopy[i*columns_d+j+1]))
			{
				matrixResult[i*columns_d+j] = matrixResultCopy[i*columns_d+j+1];
				*flagCambio_d = 1;
			}
		}
	}
}

__global__ void kernelCountFigures(int *matrixResult) {

	const int i = blockIdx.y * blockDim.y + threadIdx.y;
 	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 && i<rows_d-1 &&
		j > 0 && j<columns_d-1 &&
			matrixResult[i*columns_d+j] == i*columns_d+j) {
				atomicAdd(&numBlocks_d, 1);
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

	char *matrixDataChar_d;
	int *matrixResult_d;
    char *flagCambio_d;

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
    cudaStream_t stream[nStreams];
    for(i = 0; i < nStreams; i++) {
        gpuErrorCheck( cudaStreamCreate(&stream[i]) );
    }


	const dim3 bloqShapeGpu(columnsBloqShape,rowsBloqShape);
	const dim3 gridShapeGpu(
		ceil((float) columns / columnsBloqShape),
		ceil((float) rows / rowsBloqShape)
	);

	size_t pitch,pitch3;


    gpuErrorCheck(cudaMalloc(&flagCambio_d, sizeof(char) * nStreams));
    gpuErrorCheck(cudaMallocPitch(&matrixResult_d, &pitch, rows*sizeof(int), columns * nStreams));
	gpuErrorCheck(cudaMallocPitch(&matrixDataChar_d, &pitch3, rows*sizeof(char), columns));
	//gpuErrorCheck(cudaMalloc(&matrixDataChar_d, sizeof(char) * rows * columns));

	gpuErrorCheck(cudaMemcpyToSymbolAsync(rows_d,&rows, sizeof(int),0,cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpyToSymbolAsync(columns_d,&columns, sizeof(int),0,cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpyToSymbolAsync(matrixData_d,&matrixDataChar_d, sizeof(char *)));

	//gpuErrorCheck(cudaMallocHost(&matrixDataChar, rows*(columns) * sizeof(char)));
	matrixDataChar= (char *)malloc( rows*(columns) * sizeof(char) );
	for(i = 0; i < rows * columns; i++){
		matrixDataChar[i] = matrixData[i];
	}
	/*
	gpuErrorCheck(cudaMemcpy2D(
		matrixDataChar_d,
		pitch3,
		matrixDataChar,
		rows*sizeof(char),
		rows*sizeof(char),
		columns,
		cudaMemcpyHostToDevice
	));
	*/
	gpuErrorCheck(cudaMemcpyAsync(matrixDataChar_d,matrixDataChar,
        sizeof(char) * rows * columns,cudaMemcpyHostToDevice));


	/* 3. Etiquetado inicial */
	kernelFillMatrixResult<<<gridShapeGpu, bloqShapeGpu>>>(
        &matrixResult_d[rows * columns * (nStreams-1)]
    );
	gpuErrorCheck(cudaPeekAtLastError());

	/* 4. Computacion */
	t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	flagCambio=1;

    gpuErrorCheck(cudaMemsetAsync(flagCambio_d, 0, sizeof(char) * 4, stream[0]));
    for (i = 0; i < nStreams; i++){
        kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu,0,stream[0]>>>(
            &matrixResult_d[rows * columns * i],
            &matrixResult_d[rows * columns * ((i - 1 + nStreams) % nStreams)],
            &flagCambio_d[i]
        );
    }
    gpuErrorCheck(cudaPeekAtLastError());
    int s = -1;
    for(t=0; flagCambio != 0; t++){
		flagCambio = 0;
        s = t % nStreams;
        gpuErrorCheck(cudaMemcpyAsync(&flagCambio,&flagCambio_d[s], sizeof(char),
            cudaMemcpyDeviceToHost,stream[s]));
        gpuErrorCheck(cudaMemcpyAsync(&flagCambio_d[s],&zero, sizeof(char),
            cudaMemcpyHostToDevice,stream[s]));
        kernelComputationLoop<<<gridShapeGpu, bloqShapeGpu,0,stream[s]>>>(
            &matrixResult_d[rows * columns * s],
            &matrixResult_d[rows * columns * ((s - 1 + nStreams) % nStreams)],
            &flagCambio_d[s]
        );
	}



	/* 4.3 Inicio cuenta del numero de bloques */
	numBlocks = 0;
	gpuErrorCheck(cudaMemcpyToSymbolAsync(numBlocks_d,&numBlocks,
        sizeof(int),0,cudaMemcpyHostToDevice));
    kernelCountFigures<<<gridShapeGpu, bloqShapeGpu>>>(
        &matrixResult_d[rows * columns * (t % nStreams)]
    );
	gpuErrorCheck(cudaPeekAtLastError());
	gpuErrorCheck(cudaMemcpyFromSymbolAsync(&numBlocks, numBlocks_d,
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
	gpuErrorCheck(cudaFree(matrixData_d));
	gpuErrorCheck(cudaFree(matrixDataChar_d));
	gpuErrorCheck(cudaFree(matrixResult_d));
    for(i = 0; i < nStreams; i++) {
        gpuErrorCheck( cudaStreamDestroy(stream[i]) );
    }

	/*Liberamos los hilos del DEVICE*/
	gpuErrorCheck(cudaDeviceReset());
	return 0;
}
