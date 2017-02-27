#
# Computación paralela (Grado Ingeniería Informática)
#
# Makefile: Contar cuerpos celestes
#

CC=gcc
CFLAGS=-O3

EXES=ScanSky_seq ScanSky_seq_debug ScanSky_seq_write

all: $(EXES)

clean: 
	rm -f $(EXES)

ScanSky_seq: ScanSky_seq.c cputils.h
	$(CC) $(CFLAGS) $< -o $@

ScanSky_seq_debug: ScanSky_seq.c cputils.h
	$(CC) $(CFLAGS) -DDEBUG -DWRITE $< -o $@

ScanSky_seq_write: ScanSky_seq.c cputils.h
	$(CC) $(CFLAGS) -DWRITE $< -o  $@

