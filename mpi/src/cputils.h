/**
 * Computación Paralela
 * Funciones para las prácticas
 * 
 * @author Javier Fresno
 * @author Arturo Gonzalez-Escribano
 * @version 1.6
 *
 */
#ifndef _CPUTILS_
#define _CPUTILS_

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef CP_TABLON
#include "cputilstablon.h"
#else
#define	cp_abrir_fichero(name) fopen(name,"r")
#endif


/*
 * FUNCIONES
 */


/**
 * Función que devuelve el tiempo
 */
inline double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


#endif
