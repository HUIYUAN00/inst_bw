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
static int print_all_ranks = 0;
static int print_indices = 0;
static uint64_t nnz_count = 0;
static uint64_t *row_ptr = NULL;
static int32_t *col_idx = NULL;
static double *values = NULL;
static double *vector = NULL;
static double *result = NULL;
static double *result_ref = NULL;
static unsigned int random_seed = 42;

typedef struct {
    const char *name;
    const char *category;
    void (*func)(void *result, void *values, void *vector, uint64_t size, double scalar);
} test_item_t;

static inline double get_mflops(uint64_t flops, double time_sec) {
    return (double)flops / time_sec / 1e6;
}

static int compare_int32(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return (ia < ib) ? -1 : (ia > ib) ? 1 : 0;
}

typedef struct {
    uint32_t row;
    uint32_t col;
} sparse_coord_t;

static int compare_coord(const void *a, const void *b) {
    const sparse_coord_t *ca = (const sparse_coord_t *)a;
    const sparse_coord_t *cb = (const sparse_coord_t *)b;
    if (ca->row != cb->row) return (ca->row < cb->row) ? -1 : 1;
    return (ca->col < cb->col) ? -1 : (ca->col > cb->col) ? 1 : 0;
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

static void spmv_csr_scalar(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    double *val = (double *)values_ptr;
    double *vec = (double *)vector_ptr;
    double *y = (double *)result_ptr;
    
    for (uint64_t i = 0; i < matrix_size; i++) {
        double sum = 0.0;
        for (uint64_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += val[j] * vec[col_idx[j]];
        }
        y[i] = sum;
    }
}

static void spmv_csr_sve(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    double *val = (double *)values_ptr;
    double *vec = (double *)vector_ptr;
    double *y = (double *)result_ptr;
    
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    
    for (uint64_t i = 0; i < matrix_size; i++) {
        uint64_t row_start = row_ptr[i];
        uint64_t row_nnz = row_ptr[i + 1] - row_start;
        
        if (row_nnz == 0) {
            y[i] = 0.0;
            continue;
        }
        
        double sum = 0.0;
        for (uint64_t j = 0; j < row_nnz; j++) {
            sum += val[row_start + j] * vec[col_idx[row_start + j]];
        }
        y[i] = sum;
    }
}

#pragma GCC pop_options

static test_item_t test_registry[] = {
    {"CSR Scalar SpMV",          "SpMV",       spmv_csr_scalar},
    {"CSR SVE Intrin SpMV",      "SpMV",       spmv_csr_sve},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -M, --matrix-size <N>   Matrix dimension M (M x M) (default: 1024)\n");
    printf("  -s, --sparsity <ratio>  Sparsity ratio 0.0-1.0 (default: 0.01)\n");
    printf("  -r, --random-seed <N>   Random seed (default: 42)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -p, --print-all         Print all ranks' results (MPI only)\n");
    printf("  -i, --print-indices     Print sparse matrix indices\n");
    printf("\nExamples:\n");
    printf("  %s                             1024x1024 matrix, 1%% sparsity\n", prog_name);
    printf("  %s -M 4096 -s 0.001            4096x4096 matrix, 0.1%% sparsity\n", prog_name);
}

static void print_indices_to_file(sparse_coord_t *coords, uint64_t nnz_count, uint64_t matrix_size, 
                                   uint64_t *row_ptr, int32_t *col_idx, double *values, double sparsity) {
    char filename[256];
    snprintf(filename, sizeof(filename), "sparse_matrix_%lu_%04f.txt", matrix_size, sparsity);
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "Sparse Matrix Configuration\n");
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "Matrix Size: %lu x %lu\n", matrix_size, matrix_size);
    fprintf(fp, "Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
    fprintf(fp, "Non-zero Elements (NNZ): %lu\n", nnz_count);
    fprintf(fp, "================================================================================\n\n");
    
    fprintf(fp, "Sparse Matrix Indices (Row-Major Order):\n");
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "%-8s %-8s %-8s %-12s\n", "Index", "Row", "Col", "Value");
    fprintf(fp, "================================================================================\n");
    for (uint64_t i = 0; i < nnz_count; i++) {
        fprintf(fp, "%-8lu %-8u %-8u %-12.6f\n", i, coords[i].row, coords[i].col, values[i]);
    }
    fprintf(fp, "================================================================================\n\n");
    
    fprintf(fp, "CSR Format:\n");
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "row_ptr[%lu]:\n", matrix_size + 1);
    for (uint64_t i = 0; i <= matrix_size; i++) {
        fprintf(fp, "%lu ", row_ptr[i]);
    }
    fprintf(fp, "\n\ncol_idx[%lu]:\n", nnz_count);
    for (uint64_t i = 0; i < nnz_count; i++) {
        fprintf(fp, "%d ", col_idx[i]);
    }
    fprintf(fp, "\n\nvalues[%lu]:\n", nnz_count);
    for (uint64_t i = 0; i < nnz_count; i++) {
        fprintf(fp, "%.6f ", values[i]);
    }
    fprintf(fp, "\n================================================================================\n");
    
    fclose(fp);
    printf("Indices written to file: %s\n", filename);
}

static void print_tests(void) {
    printf("Available Tests:\n");
    printf("================================================================================\n");
    printf("%-4s %-38s %14s\n", "Idx", "Test Name", "Category");
    printf("================================================================================\n");
    for (int i = 0; i < test_count; i++) {
        printf("%-4d %-38s %14s\n", i, test_registry[i].name, test_registry[i].category);
    }
    printf("================================================================================\n");
}

static int should_run_test(int test_idx, int num_specs, char **specs) {
    if (num_specs == 0) return 1;
    
    test_item_t *test = &test_registry[test_idx];
    
    for (int i = 0; i < num_specs; i++) {
        char *spec = specs[i];
        
        char *endptr;
        long idx = strtol(spec, &endptr, 10);
        if (*endptr == '\0' && idx >= 0 && idx < test_count) {
            if (idx == test_idx) return 1;
            continue;
        }
        
        if (strcmp(spec, test->name) == 0) return 1;
        if (strcmp(spec, test->category) == 0) return 1;
        if (strstr(test->name, spec) != NULL) return 1;
    }
    return 0;
}

static int verify_spmv(void *result_ptr, void *ref_ptr) {
    double *y = (double *)result_ptr;
    double *y_ref = (double *)ref_ptr;
    int errors = 0;
    
    for (uint64_t i = 0; i < matrix_size && errors < 5; i++) {
        if (fabs(y[i] - y_ref[i]) > 1e-9) {
            if (errors == 0) fprintf(stderr, "SpMV verify FAILED:\n");
            fprintf(stderr, "  result[%lu]: expected %.6f, got %.6f\n", i, y_ref[i], y[i]);
            errors++;
        }
    }
    return errors;
}

static double run_test(test_item_t *test, void *result, void *values, void *vector
#ifdef USE_MPI
    , MPI_Comm comm
#endif
) {
    struct timespec start, end;
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    for (int i = 0; i < warmup_iter; i++) {
        test->func(result, values, vector, nnz_count, 1.0);
    }
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < test_iter; i++) {
        test->func(result, values, vector, nnz_count, 1.0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
#ifdef USE_MPI
    MPI_Barrier(comm);
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
    
    int run_all = 1;
    int num_specs = 0;
    char **specs = NULL;
    
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
            if (i + 1 < argc) {
                sparsity = atof(argv[++i]);
                if (sparsity <= 0.0) sparsity = 0.01;
                if (sparsity > 1.0) sparsity = 1.0;
            }
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
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print-all") == 0) {
            print_all_ranks = 1;
            continue;
        }
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--print-indices") == 0) {
            print_indices = 1;
            continue;
        }
        run_all = 0;
        num_specs++;
    }
    
    if (!run_all && num_specs > 0) {
        specs = &argv[argc - num_specs];
    }
    
#ifdef USE_MPI
    MPI_Bcast(&matrix_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    nnz_count = (uint64_t)(matrix_size * matrix_size * sparsity);
    if (nnz_count < matrix_size) nnz_count = matrix_size;
    
    uint64_t vl = svcntb();
    
    if (rank == 0) {
        printf("================================================================================\n");
#ifdef USE_MPI
        printf("CSR SpMV Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("CSR SpMV Benchmark\n");
#endif
        printf("================================================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("Matrix Size: %lu x %lu\n", matrix_size, matrix_size);
        printf("Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("Non-zero Elements (NNZ): %lu\n", nnz_count);
        printf("Avg NNZ per Row: %.2f\n", (double)nnz_count / matrix_size);
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Random Seed: %u\n", random_seed);
        printf("Registered Tests: %d\n", test_count);
        printf("\n");
    }
    
    if (posix_memalign((void**)&row_ptr, 64, (matrix_size + 1) * sizeof(uint64_t)) != 0 ||
        posix_memalign((void**)&col_idx, 64, nnz_count * sizeof(int32_t)) != 0 ||
        posix_memalign((void**)&values, 64, nnz_count * sizeof(double)) != 0 ||
        posix_memalign((void**)&vector, 64, matrix_size * sizeof(double)) != 0 ||
        posix_memalign((void**)&result, 64, matrix_size * sizeof(double)) != 0 ||
        posix_memalign((void**)&result_ref, 64, matrix_size * sizeof(double)) != 0) {
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
        values[i] = (double)rand() / RAND_MAX;
    }
    
    for (uint64_t i = 0; i < matrix_size; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
    
    sparse_coord_t *coords = (sparse_coord_t *)malloc(nnz_count * sizeof(sparse_coord_t));
    uint64_t *coverage = (uint64_t *)calloc(((matrix_size * matrix_size) / 64) + 2, sizeof(uint64_t));
    uint64_t unique_count = 0;
    uint64_t max_elements = matrix_size * matrix_size;
    int attempts = 0;
    int max_attempts = nnz_count * 20;
    
    while (unique_count < nnz_count && attempts < max_attempts) {
        uint64_t rand_val = ((uint64_t)rand() << 32) | (uint64_t)rand();
        uint64_t idx = rand_val % max_elements;
        uint64_t bucket = idx / 64;
        uint64_t bit = idx % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            coverage[bucket] |= (1ULL << bit);
            coords[unique_count].row = (uint32_t)(idx / matrix_size);
            coords[unique_count].col = (uint32_t)(idx % matrix_size);
            unique_count++;
        }
        attempts++;
    }
    
    for (uint64_t i = 0; i < max_elements && unique_count < nnz_count; i++) {
        uint64_t bucket = i / 64;
        uint64_t bit = i % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            coverage[bucket] |= (1ULL << bit);
            coords[unique_count].row = (uint32_t)(i / matrix_size);
            coords[unique_count].col = (uint32_t)(i % matrix_size);
            unique_count++;
        }
    }
    
    free(coverage);
    
    qsort(coords, nnz_count, sizeof(sparse_coord_t), compare_coord);
    
    row_ptr[0] = 0;
    uint64_t current_row = 0;
    
    for (uint64_t i = 0; i < nnz_count; i++) {
        col_idx[i] = (int32_t)coords[i].col;
        while (current_row < coords[i].row) {
            row_ptr[current_row + 1] = i;
            current_row++;
        }
    }
    while (current_row < matrix_size) {
        row_ptr[current_row + 1] = nnz_count;
        current_row++;
    }
    
    if (print_indices && rank == 0) {
        print_indices_to_file(coords, nnz_count, matrix_size, row_ptr, col_idx, values, sparsity);
    }
    
    free(coords);
    
    spmv_csr_scalar(result_ref, values, vector, nnz_count, 1.0);
    
    if (rank == 0) {
#ifdef USE_MPI
        printf("%-38s %14s %12s %12s %12s\n", 
               "Test", "Category", "MFLOPS", "Time(ms)", "Total(MFLOPS)");
#else
        printf("%-38s %14s %12s %12s\n", 
               "Test", "Category", "MFLOPS", "Time(ms)");
#endif
        printf("================================================================================\n");
    }
    
    for (int i = 0; i < test_count; i++) {
        if (!run_all && !should_run_test(i, num_specs, specs)) continue;
        
        test_item_t *test = &test_registry[i];
        uint64_t flops = nnz_count * 2;
        
#ifdef USE_MPI
        double time_sec = run_test(test, result, values, vector, MPI_COMM_WORLD);
#else
        double time_sec = run_test(test, result, values, vector);
#endif
        double mflops = get_mflops(flops, time_sec);
        
#ifdef USE_MPI
        double total_mflops = 0.0;
        MPI_Reduce(&mflops, &total_mflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
        
        int verify_result = verify_spmv(result, result_ref);
        
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        
        if (rank == 0 || print_all_ranks) {
#ifdef USE_MPI
            if (print_all_ranks) {
                printf("[Rank %d] %-38s %14s %12.2f %12.3f",
                       rank, test->name, test->category, mflops, time_sec * 1000);
            } else {
                printf("%-38s %14s %12.2f %12.3f %12.2f",
                       test->name, test->category, mflops, time_sec * 1000, total_mflops);
            }
#else
            printf("%-38s %14s %12.2f %12.3f",
                   test->name, test->category, mflops, time_sec * 1000);
#endif
            if (verify_result > 0) {
                printf("  VERIFY_FAIL(%d)", verify_result);
            } else if (verify_result == 0) {
                printf("  PASS");
            }
            printf("\n");
        }
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    if (rank == 0) {
        printf("================================================================================\n");
    }
    
    free(row_ptr);
    free(col_idx);
    free(values);
    free(vector);
    free(result);
    free(result_ref);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}