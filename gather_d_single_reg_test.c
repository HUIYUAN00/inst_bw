#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <arm_sve.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static int warmup_iter = 5;
static int test_iter = 10;
static uint64_t buffer_size = 0;
static double sparsity = 0.0;
static int print_all_ranks = 0;
static uint64_t num_nonzero = 0;

static double sve_gather_d_vec_idx_fmla_single_reg(void *a, void *b, int32_t *indices, uint64_t size) {
    double *dense_vec_d = (double *)a;
    double *sparse_vec_d = (double *)b;
    uint64_t vl_d = svcntd();
    uint64_t iterations = num_nonzero / vl_d;
    
    svbool_t pg = svptrue_b64();
    svfloat64_t dot_acc = svdup_f64(0.0);
    int32_t *idx_ptr = indices;
    
    for (uint64_t i = 0; i < iterations; i++) {
        svfloat64_t dense_vec = svld1_f64(pg, dense_vec_d);
        svint64_t indices_vec = svld1sw_s64(pg, idx_ptr);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, sparse_vec_d, indices_vec);
        
        dot_acc = svmla_f64_z(pg, dot_acc, dense_vec, gathered);
        
        dense_vec_d += vl_d;
        idx_ptr += vl_d;
    }
    
    return svaddv_f64(pg, dot_acc);
}

typedef struct {
    const char *name;
    double (*func)(void *a, void *b, int32_t *indices, uint64_t size);
} test_item_t;

static test_item_t test_registry[] = {
    {"SVE Gather D Vec+Idx+FMLA (Single-Reg)",   sve_gather_d_vec_idx_fmla_single_reg},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s -n <num_nonzero> -s <sparsity> [options]\n", prog_name);
    printf("\nRequired:\n");
    printf("  -n, --num-nonzero <N>   Number of non-zero elements\n");
    printf("  -s, --sparsity <ratio>  Sparsity ratio (0.0-1.0]\n");
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -p, --print-all         Print all ranks' results (MPI only)\n");
    printf("\nExamples:\n");
    printf("  %s -n 1000000 -s 0.01              1M non-zero elements, 1%% sparsity (100MB buffer)\n", prog_name);
    printf("  %s -n 500000 -s 0.02               500K non-zero elements, 2%% sparsity (25MB buffer)\n", prog_name);
    printf("  %s -n 100000 -s 0.001              100K non-zero elements, 0.1%% sparsity\n", prog_name);
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
            if (rank == 0) {
                print_usage(argv[0]);
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        }
        if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num-nonzero") == 0) {
            if (i + 1 < argc) {
                num_nonzero = (uint64_t)atol(argv[++i]);
            }
            continue;
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sparsity") == 0) {
            if (i + 1 < argc) {
                sparsity = atof(argv[++i]);
            }
            continue;
        }
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 < argc) {
                warmup_iter = atoi(argv[++i]);
            }
            continue;
        }
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            if (i + 1 < argc) {
                test_iter = atoi(argv[++i]);
            }
            continue;
        }
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print-all") == 0) {
            print_all_ranks = 1;
            continue;
        }
    }
    
    if (num_nonzero == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: -n/--num-nonzero is required and must be > 0\n");
            print_usage(argv[0]);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    if (sparsity <= 0.0 || sparsity > 1.0) {
        if (rank == 0) {
            fprintf(stderr, "Error: -s/--sparsity is required and must be in range (0.0, 1.0]\n");
            print_usage(argv[0]);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
#ifdef USE_MPI
    MPI_Bcast(&num_nonzero, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    uint64_t vl = svcntb();
    uint64_t vl_d = svcntd();
    
    num_nonzero = (num_nonzero / vl_d) * vl_d;
    if (num_nonzero == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: num_nonzero too small, must be >= %lu (VL)\n", vl_d);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    uint64_t total_elements = (uint64_t)(num_nonzero / sparsity);
    buffer_size = total_elements * sizeof(double);
    
    if (rank == 0) {
        printf("================================================================================\n");
#ifdef USE_MPI
        printf("SVE Gather D Single-Register Bandwidth Test (MPI - %d processes)\n", nprocs);
#else
        printf("SVE Gather D Single-Register Bandwidth Test\n");
#endif
        printf("================================================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("VL (double): %lu elements\n", vl_d);
        printf("Non-Zero Elements: %lu\n", num_nonzero);
        printf("Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("Calculated Buffer Size: %lu MB per array\n", buffer_size / (1024 * 1024));
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Registered Tests: %d\n", test_count);
        printf("\n");
    }
    
    void *a = NULL, *b = NULL;
    int32_t *gather_indices = NULL;
    
    a = malloc(num_nonzero * sizeof(double));
    b = malloc(buffer_size);
    gather_indices = malloc(num_nonzero * sizeof(int32_t));
    
    if (a == NULL || b == NULL || gather_indices == NULL) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate memory\n");
#endif
        return 1;
    }
    
    srand((unsigned int)time(NULL));
    
    double *da = (double *)a;
    for (uint64_t i = 0; i < num_nonzero; i++) {
        da[i] = (double)rand() / RAND_MAX;
    }
    
    uint64_t max_element_idx_64 = buffer_size / sizeof(int64_t) - 1;
    uint64_t max_idx = (max_element_idx_64 < INT32_MAX) ? max_element_idx_64 : INT32_MAX;
    
    uint64_t stride = (max_idx + 1) / num_nonzero;
    if (stride == 0) stride = 1;
    for (uint64_t i = 0; i < num_nonzero; i++) {
        uint64_t idx = i * stride;
        if (idx > max_idx) idx = max_idx;
        gather_indices[i] = (int32_t)idx;
    }
    
    memset(b, 0, buffer_size);
    double *db = (double *)b;
    for (uint64_t i = 0; i < num_nonzero; i++) {
        db[gather_indices[i]] = (double)rand() / RAND_MAX;
    }
    
    if (rank == 0) {
        printf("Index Mode: Uniform\n");
        printf("Max Index: %lu (buffer elements: %lu)\n", max_idx, max_element_idx_64);
        printf("Stride: %lu\n", stride);
        printf("\n");
    }
    
    if (rank == 0) {
#ifdef USE_MPI
        printf("%-38s %10s %10s %10s\n", 
               "Test", "MFLOPS", "Time(ms)", "Total(MFLOPS)");
#else
        printf("%-38s %10s %10s\n", 
               "Test", "MFLOPS", "Time(ms)");
#endif
        printf("================================================================================\n");
    }
    
    struct timeval start, end;
    double last_result = 0.0;
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    for (int w = 0; w < warmup_iter; w++) {
        test_registry[0].func(a, b, gather_indices, buffer_size);
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    gettimeofday(&start, NULL);
    for (int t = 0; t < test_iter; t++) {
        last_result = test_registry[0].func(a, b, gather_indices, buffer_size);
    }
    gettimeofday(&end, NULL);
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    double total_time_sec = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    double time_per_iter = total_time_sec / test_iter;
    
    test_item_t *test = &test_registry[0];
    double mflops = (double)num_nonzero * 2.0 / time_per_iter / 1e6;
    
#ifdef USE_MPI
    double total_mflops = 0.0;
    MPI_Reduce(&mflops, &total_mflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    
    if (rank == 0 || print_all_ranks) {
#ifdef USE_MPI
        if (print_all_ranks) {
            printf("[Rank %d] %-38s %10.2f %10.3f",
                   rank, test->name, mflops, time_per_iter * 1000);
        } else {
            printf("%-38s %10.2f %10.3f",
                   test->name, mflops, time_per_iter * 1000);
        }
#else
        printf("%-38s %10.2f %10.3f",
               test->name, mflops, time_per_iter * 1000);
#endif
        
#ifdef USE_MPI
        if (!print_all_ranks) {
            printf(" %10.2f", total_mflops);
        }
#endif
        printf("\n");
        
        if (rank == 0) {
            printf("  Final dot product result: %.15e\n", last_result);
        }
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    if (rank == 0) {
        printf("================================================================================\n");
    }
    
    free(a);
    free(b);
    free(gather_indices);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}