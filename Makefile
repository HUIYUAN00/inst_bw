CC = gcc
MPICC = mpicc
CFLAGS = -O3 -march=armv8-a+sve -mtune=native -Wall
LDFLAGS = -lm
MPI_CFLAGS = $(CFLAGS) -DUSE_MPI
OMP_CFLAGS = $(CFLAGS) -fopenmp

all: sve_bw_test sve_bw_test_mpi gather_scatter_test gather_scatter_test_mpi stream_omp_test

sve_bw_test: sve_bw_test.c
	$(CC) $(CFLAGS) -o $@ $<

sve_bw_test_mpi: sve_bw_test.c
	$(MPICC) $(MPI_CFLAGS) -o $@ $<

gather_scatter_test: gather_scatter_test.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

gather_scatter_test_mpi: gather_scatter_test.c
	$(MPICC) $(MPI_CFLAGS) $(LDFLAGS) -o $@ $<

stream_omp_test: stream_omp_test.c
	$(CC) $(OMP_CFLAGS) -o $@ $<

clean:
	rm -f sve_bw_test sve_bw_test_mpi gather_scatter_test gather_scatter_test_mpi stream_omp_test

run: sve_bw_test_mpi
	mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi

run_single: sve_bw_test
	./sve_bw_test

run_gs: gather_scatter_test
	./gather_scatter_test

run_gs_mpi: gather_scatter_test_mpi
	mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi

run_stream_omp: stream_omp_test
	./stream_omp_test -n 4

run_stream_omp_8: stream_omp_test
	./stream_omp_test -n 8

.PHONY: all clean run run_single run_gs run_gs_mpi run_stream_omp run_stream_omp_8