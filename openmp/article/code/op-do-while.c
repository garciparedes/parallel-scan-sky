/* 4. Computacion */
do {
    #pragma omp barrier

    #pragma omp single
    flagCambio=0;

    //...

} while(flagCambio !=0);
