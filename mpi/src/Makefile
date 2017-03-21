#
# Computaci�n paralela (Grado Ingenier�a Inform�tica)
#
# Makefile: Contar cuerpos celestes
#

CC=mpicc
CFLAGS=-O3

EXES=ScanSky_mpi ScanSky_mpi_debug ScanSky_mpi_write

all: $(EXES)

clean:
	rm -f $(EXES)

ScanSky_mpi: ScanSky_mpi.c cputils.h
	$(CC) $(CFLAGS) $< -o $@

ScanSky_mpi_debug: ScanSky_mpi.c cputils.h
	$(CC) $(CFLAGS) -DDEBUG -DWRITE $< -o $@

ScanSky_mpi_write: ScanSky_mpi.c cputils.h
	$(CC) $(CFLAGS) -DWRITE $< -o  $@
