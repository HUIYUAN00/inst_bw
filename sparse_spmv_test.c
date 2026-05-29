#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_sve.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static int warmup_iter = 5;
static int test_iter = 10;
static uint64_t matrix_size = 1024;
static double sparsity = 0.01;
static uint64_t nnz_count = 0;
static int32_t *col_indices = NULL;
static double *matrix_values = NULL;
static double *vector = NULL;
static double *result = NULL;
static unsigned int random_seed = 42;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

static int compare_int32(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return (ia < ib) ? -1 : (ia > ib) ? 1 : 0;
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

static void spmv_sve_gather(void *result_ptr, void *matrix_ptr, void *vector_ptr, uint64_t size, double scalar) {
    double *matrix = (double *)matrix_ptr;
    double *vec = (double *)vector_ptr;
    double *dst = (double *)result_ptr;
    int32_t *idx_base = col_indices;
    
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t iterations = nnz_count / vl_d;
    if (iterations == 0) iterations = 1;
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, %[idx]\n"
        "mov x18, %[mat]\n"
        "mov x19, %[vec]\n"
        "mov x20, %[dst]\n"
        "mov x21, #0\n"
        "1:\n"
        "ptrue p0.d\n"
        "ld1sw z4.d, p0/z, [x17, x21, lsl 2]\n"
        "ld1d z0.d, p0/z, [x18, x21, lsl 3]\n"
        "ld1d z1.d, p0/z, [x19, z4.d, lsl 3]\n"
        "fmla z2.d, p0/m, z0.d, z1.d\n"
        "add x21, x21, %[vl]\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        "st1d z2.d, p0, [x20]\n"
        :
        : [iter] "r" (iterations), [idx] "r" (idx_base), [mat] "r" (matrix), 
          [vec] "r" (vec), [dst] "r" (dst), [vl] "r" (vl_d)
        : "x16", "x17", "x18", "x19", "x20", "x21", "p0",
          "z0", "z1", "z2", "z4", "memory"
    );
}

static void spmv_scalar(void *result_ptr, void *matrix_ptr, void *vector_ptr, uint64_t size, double scalar) {
    double *matrix = (double *)matrix_ptr;
    double *vec = (double *)vector_ptr;
    double *dst = (double *)result_ptr;
    int32_t *idx = col_indices;
    
    double sum = 0.0;
    for (uint64_t i = 0; i < nnz_count; i++) {
        sum += matrix[i] * vec[idx[i]];
    }
    *dst = sum;
}

#pragma GCC pop_options

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -M, --matrix-size <N>   Matrix dimension M (M x M) (default: 1024)\n");
    printf("  -s, --sparsity <ratio>  Sparsity ratio 0.0-1.0 (default: 0.01)\n");
    printf("  -r, --random-seed <N>   Random seed (default: 42)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("\nExamples:\n");
    printf("  %s                             1024x1024 matrix, 1%% sparsity\n", prog_name);
    printf("  %s -M 4096 -s 0.001            4096x4096 matrix, 0.1%% sparsity\n", prog_name);
}

static int verify_spmv(void *result_ptr, void *matrix_ptr, void *vector_ptr) {
    double *matrix = (double *)matrix_ptr;
    double *vec = (double *)vector_ptr;
    double *dst = (double *)result_ptr;
    int32_t *idx = col_indices;
    
    double expected = 0.0;
    for (uint64_t i = 0; i < nnz_count; i++) {
        expected += matrix[i] * vec[idx[i]];
    }
    
    if (fabs(*dst - expected) > 1e-6) {
        fprintf(stderr, "SPMV verify FAILED: expected %.6f, got %.6f\n", expected, *dst);
        return 1;
    }
    return 0;
}

static double run_test(void (*func)(void*, void*, void*, uint64_t, double), void *result, void *matrix, void *vector) {
    struct timespec start, end;
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    for (int i = 0; i < warmup_iter; i++) {
        func(result, matrix, vector, nnz_count, 1.0);
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < test_iter; i++) {
        func(result, matrix, vector, nnz_count, 1.0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return time_sec / test_iter;
}

int main(int argc, char *argv[]) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    int rank = 0;
#endif
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            if (rank == 0) print_usage(argv[0]);
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        }
        if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) matrix_size = (uint64_t)atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sparsity") == 0) {
            if (i + 1 < argc) sparsity = atof(argv[++i]);
            if (sparsity <= 0.0) sparsity = 0.01;
            if (sparsity > 1.0) sparsity = 1.0;
            continue;
        }
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random-seed") == 0) {
            if (i + 1 < argc) random_seed = (unsigned int)atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 < argc) warmup_iter = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            if (i + 1 < argc) test_iter = atoi(argv[++i]);
            continue;
        }
    }
    
#ifdef USE_MPI
    MPI_Bcast(&matrix_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    nnz_count = (uint64_t)(matrix_size * matrix_size * sparsity);
    if (nnz_count < 1) nnz_count = 1;
    
    uint64_t vl = svcntb();
    
    if (rank == 0) {
        printf("================================================================================\n");
#ifdef USE_MPI
        printf("Sparse SpMV Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("Sparse SpMV Benchmark\n");
#endif
        printf("================================================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("Matrix Size: %lu x %lu\n", matrix_size, matrix_size);
        printf("Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("Non-zero Elements (NNZ): %lu\n", nnz_count);
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Random Seed: %u\n", random_seed);
        printf("\n");
    }
    
    if (posix_memalign((void**)&matrix_values, 64, nnz_count * sizeof(double)) != 0 ||
        posix_memalign((void**)&col_indices, 64, nnz_count * sizeof(int32_t)) != 0 ||
        posix_memalign((void**)&vector, 64, matrix_size * sizeof(double)) != 0 ||
        posix_memalign((void**)&result, 64, sizeof(double)) != 0) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate aligned memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate aligned memory\n");
#endif
        return 1;
    }
    
    srand(random_seed);
    
    for (uint64_t i = 0; i < nnz_count; i++) {
        matrix_values[i] = (double)rand() / RAND_MAX;
    }
    
    for (uint64_t i = 0; i < matrix_size; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
    
    uint64_t *coverage = (uint64_t *)calloc((matrix_size / 64) + 2, sizeof(uint64_t));
    uint64_t unique_count = 0;
    int attempts = 0;
    int max_attempts = nnz_count * 20;
    
    while (unique_count < nnz_count && attempts < max_attempts) {
        uint64_t idx = ((uint64_t)rand() << 32 | rand()) % matrix_size;
        uint64_t bucket = idx / 64;
        uint64_t bit = idx % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            coverage[bucket] |= (1ULL << bit);
            col_indices[unique_count++] = (int32_t)idx;
        }
        attempts++;
    }
    
    for (uint64_t i = 0; i < matrix_size && unique_count < nnz_count; i++) {
        uint64_t bucket = i / 64;
        uint64_t bit = i % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            coverage[bucket] |= (1ULL << bit);
            col_indices[unique_count++] = (int32_t)i;
        }
    }
    
    qsort(col_indices, nnz_count, sizeof(int32_t), compare_int32);
    
    for (uint64_t i = 0; i < nnz_count; i++) {
        col_indices[i] = col_indices[i] % matrix_size;
    }
    
    free(coverage);
    
    if (rank == 0) {
        printf("%-38s %12s %12s %12s\n", "Test", "Time(ms)", "GFLOPS", "GB/s");
        printf("================================================================================\n");
    }
    
    double time_sec = run_test(spmv_sve_gather, result, matrix_values, vector);
    uint64_t flops = nnz_count * 2;
    uint64_t bytes = nnz_count * sizeof(double) + nnz_count * sizeof(int32_t) + nnz_count * sizeof(double);
    double gflops = (double)flops / time_sec / 1e9;
    double bandwidth = get_bandwidth(bytes, time_sec);
    
    int verify_result = verify_spmv(result, matrix_values, vector);
    
    if (rank == 0) {
        printf("%-38s %12.3f %12.2f %12.2f", "SVE Gather SpMV", time_sec * 1000, gflops, bandwidth);
        if (verify_result == 0) {
            printf("  PASS\n");
        } else {
            printf("  FAIL\n");
        }
    }
    
    memset(result, 0, sizeof(double));
    time_sec = run_test(spmv_scalar, result, matrix_values, vector);
    gflops = (double)flops / time_sec / 1e9;
    bandwidth = get_bandwidth(bytes, time_sec);
    
    verify_result = verify_spmv(result, matrix_values, vector);
    
    if (rank == 0) {
        printf("%-38s %12.3f %12.2f %12.2f", "Scalar SpMV", time_sec * 1000, gflops, bandwidth);
        if (verify_result == 0) {
            printf("  PASS\n");
        } else {
            printf("  FAIL\n");
        }
        printf("================================================================================\n");
    }
    
    free(matrix_values);
    free(col_indices);
    free(vector);
    free(result);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}