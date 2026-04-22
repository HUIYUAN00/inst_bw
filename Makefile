CC = gcc
MPICC = mpicc
CFLAGS = -O3 -march=armv8-a+sve -mtune=native -Wall
MPI_CFLAGS = $(CFLAGS) -DUSE_MPI

all: sve_bw_test sve_bw_test_mpi

sve_bw_test: sve_bw_test.c
	$(CC) $(CFLAGS) -o $@ $<

sve_bw_test_mpi: sve_bw_test.c
	$(MPICC) $(MPI_CFLAGS) -o $@ $<

clean:
	rm -f sve_bw_test sve_bw_test_mpi

run: sve_bw_test_mpi
	mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi

run_single: sve_bw_test
	./sve_bw_test

.PHONY: all clean run run_single