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
static int block_size = 4;
static double sparsity = 0.1;
static unsigned int random_seed = 42;
static int print_all_ranks = 0;

typedef struct {
    double re;
    double im;
} complex_double_t;

static uint64_t nnz_blocks = 0;
static uint64_t *row_ptr = NULL;
static int32_t *col_idx = NULL;
static complex_double_t *values = NULL;
static complex_double_t *vector = NULL;
static complex_double_t *result = NULL;
static complex_double_t *result_ref = NULL;

typedef struct {
    const char *name;
    const char *category;
    void (*func)(void *result, void *values, void *vector, uint64_t size, double scalar);
} test_item_t;

static inline double get_mflops(uint64_t flops, double time_sec) {
    return (double)flops / time_sec / 1e6;
}

typedef struct {
    uint32_t row;
    uint32_t col;
} block_coord_t;

static int compare_coord(const void *a, const void *b) {
    const block_coord_t *ca = (const block_coord_t *)a;
    const block_coord_t *cb = (const block_coord_t *)b;
    if (ca->row != cb->row) return (ca->row < cb->row) ? -1 : 1;
    return (ca->col < cb->col) ? -1 : (ca->col > cb->col) ? 1 : 0;
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

/*
 * BSR Hermitian SpMV: y = A*x
 * 仅存储上三角块(I<=J),利用共轭对称性计算下三角贡献。
 *
 * 对角块(I==J): 块本身为Hermitian,直接乘加到 y[I*r+i]
 * 上三角块(I<J):
 *   内积: y[I*r+i] += a * x[J*r+k]
 *   外积: y[J*r+k] += conj(a) * x[I*r+i]  (共轭转置贡献)
 */
static void spmv_bsr_herm_scalar(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    (void)size;
    (void)scalar;
    complex_double_t *val = (complex_double_t *)values_ptr;
    complex_double_t *vec = (complex_double_t *)vector_ptr;
    complex_double_t *y = (complex_double_t *)result_ptr;
    int r = block_size;
    uint64_t nbr = matrix_size / r;

    for (uint64_t i = 0; i < matrix_size; i++) {
        y[i].re = 0.0;
        y[i].im = 0.0;
    }

    for (uint64_t I = 0; I < nbr; I++) {
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            int32_t J = col_idx[j];
            complex_double_t *block = &val[j * r * r];

            for (int i = 0; i < r; i++) {
                for (int k = 0; k < r; k++) {
                    complex_double_t a = block[i * r + k];
                    complex_double_t xj = vec[J * r + k];
                    y[I * r + i].re += a.re * xj.re - a.im * xj.im;
                    y[I * r + i].im += a.re * xj.im + a.im * xj.re;
                }
            }

            if (I < J) {
                complex_double_t *xi = &vec[I * r];
                for (int k = 0; k < r; k++) {
                    for (int i = 0; i < r; i++) {
                        complex_double_t a = block[i * r + k];
                        y[J * r + k].re += a.re * xi[i].re + a.im * xi[i].im;
                        y[J * r + k].im += a.re * xi[i].im - a.im * xi[i].re;
                    }
                }
            }
        }
    }
}

static void spmv_bsr_herm_sve(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    (void)size;
    (void)scalar;
    complex_double_t *val = (complex_double_t *)values_ptr;
    complex_double_t *vec = (complex_double_t *)vector_ptr;
    complex_double_t *y = (complex_double_t *)result_ptr;
    int r = block_size;
    uint64_t nbr = matrix_size / r;
    uint64_t vl_doubles = svcntd();

    for (uint64_t i = 0; i < matrix_size; i++) {
        y[i].re = 0.0;
        y[i].im = 0.0;
    }

    for (uint64_t I = 0; I < nbr; I++) {
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            int32_t J = col_idx[j];
            complex_double_t *block = &val[j * r * r];

            for (int i = 0; i < r; i++) {
                double *br = (double *)&block[i * r];
                double *xr = (double *)&vec[J * r];
                uint64_t total = (uint64_t)r * 2;
                uint64_t off = 0;
                double sum_re = 0.0, sum_im = 0.0;

                while (off < total) {
                    uint64_t cnt = (total - off < vl_doubles) ? (total - off) : vl_doubles;
                    double pre, pim;

                    __asm__ volatile (
                        "ptrue p2.b\n\t"
                        "whilelt p1.d, xzr, %[cnt]\n\t"
                        "ld1d z0.d, p1/z, [%[br], %[off], lsl #3]\n\t"
                        "ld1d z1.d, p1/z, [%[xr], %[off], lsl #3]\n\t"
                        "mov z2.d, #0\n\t"
                        "fcmla z2.d, p1/m, z0.d, z1.d, #0\n\t"
                        "fcmla z2.d, p1/m, z0.d, z1.d, #90\n\t"
                        "mov z5.d, #0\n\t"
                        "uzp1 z3.d, z2.d, z5.d\n\t"
                        "uzp2 z4.d, z2.d, z5.d\n\t"
                        "faddv d5, p2, z3.d\n\t"
                        "faddv d6, p2, z4.d\n\t"
                        "fmov %[pre], d5\n\t"
                        "fmov %[pim], d6\n\t"
                        : [pre] "=r"(pre), [pim] "=r"(pim)
                        : [br] "r"(br), [xr] "r"(xr), [off] "r"(off), [cnt] "r"(cnt)
                        : "p1", "p2", "z0", "z1", "z2", "z3", "z4", "z5", "d5", "d6", "memory"
                    );

                    sum_re += pre;
                    sum_im += pim;
                    off += cnt;
                }

                y[I * r + i].re += sum_re;
                y[I * r + i].im += sum_im;
            }

            if (I < J) {
                complex_double_t *xi = &vec[I * r];
                for (int k = 0; k < r; k++) {
                    for (int i = 0; i < r; i++) {
                        complex_double_t a = block[i * r + k];
                        y[J * r + k].re += a.re * xi[i].re + a.im * xi[i].im;
                        y[J * r + k].im += a.re * xi[i].im - a.im * xi[i].re;
                    }
                }
            }
        }
    }
}

#pragma GCC pop_options

static test_item_t test_registry[] = {
    {"BSR Herm Scalar",     "HEMV", spmv_bsr_herm_scalar},
    {"BSR Herm SVE",        "HEMV", spmv_bsr_herm_sve},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -M, --matrix-size <N>   Matrix dimension N (N x N) (default: 1024)\n");
    printf("  -b, --block-size <r>    Block size r (r x r) (default: 4)\n");
    printf("  -s, --sparsity <ratio>  Block sparsity ratio 0.0-1.0 (default: 0.1)\n");
    printf("  -r, --random-seed <N>   Random seed (default: 42)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -p, --print-all         Print all ranks' results (MPI only)\n");
    printf("\nNote: Tests Hermitian BSR SpMV with upper triangle block storage\n");
    printf("      Diagonal blocks are Hermitian (conjugate-symmetric)\n");
    printf("      Off-diagonal blocks use conjugate symmetry for lower triangle\n");
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

static int verify_hemv(void *result_ptr, void *ref_ptr) {
    complex_double_t *y = (complex_double_t *)result_ptr;
    complex_double_t *y_ref = (complex_double_t *)ref_ptr;
    int errors = 0;

    for (uint64_t i = 0; i < matrix_size && errors < 5; i++) {
        double re_diff = fabs(y[i].re - y_ref[i].re);
        double im_diff = fabs(y[i].im - y_ref[i].im);
        if (re_diff > 1e-9 || im_diff > 1e-9) {
            if (errors == 0) fprintf(stderr, "BSR HEMV verify FAILED:\n");
            fprintf(stderr, "  result[%lu]: expected (%.6f,%.6f), got (%.6f,%.6f)\n",
                    i, y_ref[i].re, y_ref[i].im, y[i].re, y[i].im);
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
        test->func(result, values, vector, nnz_blocks, 1.0);
    }

#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < test_iter; i++) {
        test->func(result, values, vector, nnz_blocks, 1.0);
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
        if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            if (rank == 0) print_tests();
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        }
        if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) matrix_size = (uint64_t)atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 < argc) block_size = atoi(argv[++i]);
            if (block_size < 1) block_size = 4;
            continue;
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sparsity") == 0) {
            if (i + 1 < argc) {
                sparsity = atof(argv[++i]);
                if (sparsity <= 0.0) sparsity = 0.1;
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
        run_all = 0;
        num_specs++;
    }

    if (!run_all && num_specs > 0) {
        specs = &argv[argc - num_specs];
    }

    if (matrix_size % block_size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: matrix_size (%lu) must be divisible by block_size (%d)\n",
                    matrix_size, block_size);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

#ifdef USE_MPI
    MPI_Bcast(&matrix_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    uint64_t num_block_rows = matrix_size / block_size;
    uint64_t total_upper_blocks = num_block_rows * (num_block_rows + 1) / 2;
    nnz_blocks = (uint64_t)(total_upper_blocks * sparsity);
    if (nnz_blocks < num_block_rows) nnz_blocks = num_block_rows;
    if (nnz_blocks > total_upper_blocks) nnz_blocks = total_upper_blocks;

    uint64_t vl = svcntb();
    uint64_t nnz_elements = nnz_blocks * block_size * block_size;

    if (rank == 0) {
        printf("================================================================================\n");
#ifdef USE_MPI
        printf("BSR HEMV Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("BSR HEMV Benchmark\n");
#endif
        printf("================================================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("Matrix Size: %lu x %lu\n", matrix_size, matrix_size);
        printf("Block Size: %d x %d\n", block_size, block_size);
        printf("Num Block Rows: %lu\n", num_block_rows);
        printf("Upper Triangle Blocks: %lu\n", total_upper_blocks);
        printf("Block Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("NNZ Blocks: %lu\n", nnz_blocks);
        printf("NNZ Elements: %lu\n", nnz_elements);
        printf("Avg NNZ Blocks per Row: %.2f\n", (double)nnz_blocks / num_block_rows);
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Random Seed: %u\n", random_seed);
        printf("Registered Tests: %d\n", test_count);
        printf("\n");
    }

    if (posix_memalign((void**)&row_ptr, 64, (num_block_rows + 1) * sizeof(uint64_t)) != 0 ||
        posix_memalign((void**)&col_idx, 64, nnz_blocks * sizeof(int32_t)) != 0 ||
        posix_memalign((void**)&values, 64, nnz_elements * sizeof(complex_double_t)) != 0 ||
        posix_memalign((void**)&vector, 64, matrix_size * sizeof(complex_double_t)) != 0 ||
        posix_memalign((void**)&result, 64, matrix_size * sizeof(complex_double_t)) != 0 ||
        posix_memalign((void**)&result_ref, 64, matrix_size * sizeof(complex_double_t)) != 0) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate aligned memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate aligned memory\n");
#endif
        return 1;
    }

    srand(random_seed);

    for (uint64_t i = 0; i < matrix_size; i++) {
        vector[i].re = (double)rand() / RAND_MAX;
        vector[i].im = (double)rand() / RAND_MAX;
    }

    block_coord_t *coords = (block_coord_t *)malloc(nnz_blocks * sizeof(block_coord_t));
    uint64_t *coverage = (uint64_t *)calloc((total_upper_blocks / 64) + 2, sizeof(uint64_t));
    uint64_t unique_count = 0;
    int attempts = 0;
    int max_attempts = nnz_blocks * 20;

    while (unique_count < nnz_blocks && attempts < max_attempts) {
        uint64_t rand_val = ((uint64_t)rand() << 32) | (uint64_t)rand();
        uint64_t I = rand_val % num_block_rows;
        uint64_t J = rand_val % num_block_rows;
        if (I > J) { uint64_t t = I; I = J; J = t; }
        uint64_t idx = I * num_block_rows - I * (I + 1) / 2 + J;
        uint64_t bucket = idx / 64;
        uint64_t bit = idx % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            coverage[bucket] |= (1ULL << bit);
            coords[unique_count].row = (uint32_t)I;
            coords[unique_count].col = (uint32_t)J;
            unique_count++;
        }
        attempts++;
    }

    for (uint64_t I = 0; I < num_block_rows && unique_count < nnz_blocks; I++) {
        for (uint64_t J = I; J < num_block_rows && unique_count < nnz_blocks; J++) {
            uint64_t idx = I * num_block_rows - I * (I + 1) / 2 + J;
            uint64_t bucket = idx / 64;
            uint64_t bit = idx % 64;
            if (!(coverage[bucket] & (1ULL << bit))) {
                coverage[bucket] |= (1ULL << bit);
                coords[unique_count].row = (uint32_t)I;
                coords[unique_count].col = (uint32_t)J;
                unique_count++;
            }
        }
    }

    free(coverage);

    qsort(coords, nnz_blocks, sizeof(block_coord_t), compare_coord);

    row_ptr[0] = 0;
    uint64_t current_row = 0;

    for (uint64_t i = 0; i < nnz_blocks; i++) {
        col_idx[i] = (int32_t)coords[i].col;
        while (current_row < coords[i].row) {
            row_ptr[current_row + 1] = i;
            current_row++;
        }
    }
    while (current_row < num_block_rows) {
        row_ptr[current_row + 1] = nnz_blocks;
        current_row++;
    }

    int r = block_size;
    for (uint64_t i = 0; i < nnz_blocks; i++) {
        uint64_t I = coords[i].row;
        uint64_t J = coords[i].col;
        complex_double_t *block = &values[i * r * r];

        if (I == J) {
            for (int bi = 0; bi < r; bi++) {
                for (int bk = 0; bk < r; bk++) {
                    if (bi == bk) {
                        block[bi * r + bk].re = (double)rand() / RAND_MAX;
                        block[bi * r + bk].im = 0.0;
                    } else if (bi < bk) {
                        double re = (double)rand() / RAND_MAX;
                        double im = (double)rand() / RAND_MAX;
                        block[bi * r + bk].re = re;
                        block[bi * r + bk].im = im;
                        block[bk * r + bi].re = re;
                        block[bk * r + bi].im = -im;
                    }
                }
            }
        } else {
            for (int bi = 0; bi < r; bi++) {
                for (int bk = 0; bk < r; bk++) {
                    block[bi * r + bk].re = (double)rand() / RAND_MAX;
                    block[bi * r + bk].im = (double)rand() / RAND_MAX;
                }
            }
        }
    }

    free(coords);

    spmv_bsr_herm_scalar(result_ref, values, vector, nnz_blocks, 1.0);

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
        uint64_t flops = nnz_elements * 8;

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

        int verify_result = verify_hemv(result, result_ref);

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
