CC = gcc
MPICC = mpicc
CFLAGS = -O3 -fPIC -msve-vector-bits=256 -march=armv8.3-a+sve -Wall
LDFLAGS = -lm
MPI_CFLAGS = $(CFLAGS) -DUSE_MPI
OMP_CFLAGS = $(CFLAGS) -fopenmp

all: sve_bw_test sve_bw_test_mpi gather_scatter_test gather_scatter_test_mpi stream_omp_test sparse_spmv_test sparse_spmv_test_mpi gather_d_single_reg_test gather_d_single_reg_test_mpi matrix_scale_test matrix_scale_test_mpi hermitian_spmv_test hermitian_spmv_test_mpi bsr_spmv_test bsr_spmv_test_mpi test_spmv_asm

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

matrix_scale_test: matrix_scale_test.c
	$(CC) $(CFLAGS) -o $@ $<

matrix_scale_test_mpi: matrix_scale_test.c
	$(MPICC) $(MPI_CFLAGS) -o $@ $<

hermitian_spmv_test: hermitian_spmv_test.c
	$(CC) $(CFLAGS) -o $@ $<

hermitian_spmv_test_mpi: hermitian_spmv_test.c
	$(MPICC) $(MPI_CFLAGS) -o $@ $<

bsr_spmv_test: bsr_spmv_test.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

bsr_spmv_test_mpi: bsr_spmv_test.c
	$(MPICC) $(MPI_CFLAGS) $(LDFLAGS) -o $@ $<

spmv_asm.o: spmv_asm.s
	$(CC) $(CFLAGS) -c -o $@ $<

test_spmv_asm: test_spmv_asm.c spmv_asm.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f sve_bw_test sve_bw_test_mpi gather_scatter_test gather_scatter_test_mpi stream_omp_test sparse_spmv_test sparse_spmv_test_mpi gather_d_single_reg_test gather_d_single_reg_test_mpi matrix_scale_test matrix_scale_test_mpi hermitian_spmv_test hermitian_spmv_test_mpi bsr_spmv_test bsr_spmv_test_mpi spmv_asm.o test_spmv_asm

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

run_matrix_scale: matrix_scale_test
	./matrix_scale_test

run_matrix_scale_mpi: matrix_scale_test_mpi
	mpirun --allow-run-as-root -np 4 ./matrix_scale_test_mpi

run_hermitian_spmv: hermitian_spmv_test
	./hermitian_spmv_test

run_hermitian_spmv_mpi: hermitian_spmv_test_mpi
	mpirun --allow-run-as-root -np 4 ./hermitian_spmv_test_mpi

run_spmv_asm: test_spmv_asm
	./test_spmv_asm

run_bsr_spmv: bsr_spmv_test
	./bsr_spmv_test

run_bsr_spmv_mpi: bsr_spmv_test_mpi
	mpirun --allow-run-as-root -np 4 ./bsr_spmv_test_mpi

.PHONY: all clean run run_single run_gs run_gs_mpi run_stream_omp run_stream_omp_8 run_spmv run_spmv_mpi run_gather_d_single run_gather_d_single_mpi run_matrix_scale run_matrix_scale_mpi run_hermitian_spmv run_hermitian_spmv_mpi run_spmv_asm run_bsr_spmv run_bsr_spmv_mpi