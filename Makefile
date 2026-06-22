CC = gcc
MPICC = mpicc
CFLAGS = -O3 -fPIC -msve-vector-bits=256 -march=armv8.3-a+sve -Wall
LDFLAGS = -lm
MPI_CFLAGS = $(CFLAGS) -DUSE_MPI
OMP_CFLAGS = $(CFLAGS) -fopenmp

all: sve_bw_test sve_bw_test_mpi gather_scatter_test gather_scatter_test_mpi stream_omp_test sparse_spmv_test sparse_spmv_test_mpi gather_d_single_reg_test gather_d_single_reg_test_mpi

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

sparse_spmv_test: sparse_spmv_test.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

sparse_spmv_test_mpi: sparse_spmv_test.c
	$(MPICC) $(MPI_CFLAGS) $(LDFLAGS) -o $@ $<

gather_d_single_reg_test: gather_d_single_reg_test.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

gather_d_single_reg_test_mpi: gather_d_single_reg_test.c
	$(MPICC) $(MPI_CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f sve_bw_test sve_bw_test_mpi gather_scatter_test gather_scatter_test_mpi stream_omp_test sparse_spmv_test sparse_spmv_test_mpi gather_d_single_reg_test gather_d_single_reg_test_mpi

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

run_spmv: sparse_spmv_test
	./sparse_spmv_test

run_spmv_mpi: sparse_spmv_test_mpi
	mpirun --allow-run-as-root -np 4 ./sparse_spmv_test_mpi

run_gather_d_single: gather_d_single_reg_test
	./gather_d_single_reg_test

run_gather_d_single_mpi: gather_d_single_reg_test_mpi
	mpirun --allow-run-as-root -np 4 ./gather_d_single_reg_test_mpi

.PHONY: all clean run run_single run_gs run_gs_mpi run_stream_omp run_stream_omp_8 run_spmv run_spmv_mpi run_gather_d_single run_gather_d_single_mpi