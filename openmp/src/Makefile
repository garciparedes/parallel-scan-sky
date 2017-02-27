#
# Computación paralela (Grado Ingeniería Informática)
#
# Makefile: Contar cuerpos celestes
#

CC=gcc
CFLAGS=-O3

EXES=ScanSky_openmp ScanSky_openmp_debug ScanSky_openmp_write

all: $(EXES)

clean:
	rm -f $(EXES)

ScanSky_openmp: ScanSky_openmp.c cputils.h
	$(CC) $(CFLAGS) $< -o $@

ScanSky_openmp_debug: ScanSky_openmp.c cputils.h
	$(CC) $(CFLAGS) -DDEBUG -DWRITE $< -o $@

ScanSky_openmp_write: ScanSky_openmp.c cputils.h
	$(CC) $(CFLAGS) -DWRITE $< -o  $@
