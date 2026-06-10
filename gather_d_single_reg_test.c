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
static uint64_t buffer_size = 0;
static double sparsity = 0.0;
static int index_mode = 0;
static int print_all_ranks = 0;
static uint64_t num_nonzero = 0;
static int32_t *gather_indices = NULL;
static unsigned int random_seed = 42;
static const char *output_indices_file = NULL;
static uint64_t index_modulo = 0;
static volatile double last_dot_result = 0.0;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

static int compare_uint64(const void *a, const void *b) {
    uint64_t ua = *(const uint64_t *)a;
    uint64_t ub = *(const uint64_t *)b;
    return (ua < ub) ? -1 : (ua > ub) ? 1 : 0;
}

static inline void update_index_stats(uint64_t idx, uint64_t *min_idx, uint64_t *max_found, uint64_t *coverage) {
    if (idx < *min_idx) *min_idx = idx;
    if (idx > *max_found) *max_found = idx;
    coverage[idx / 64] |= (1ULL << (idx % 64));
}

static double sve_gather_d_vec_idx_fmla_single_reg_core(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *dense_vec_d = (double *)a;
    double *sparse_vec_d = (double *)c;
    uint64_t vl_d = svcntd();
    uint64_t iterations = num_nonzero / vl_d;
    
    svbool_t pg = svptrue_b64();
    svfloat64_t dot_acc = svdup_f64(0.0);
    int32_t *idx_ptr = gather_indices;
    
    for (uint64_t i = 0; i < iterations; i++) {
        svfloat64_t dense_vec = svld1_f64(pg, dense_vec_d);
        svint64_t indices = svld1sw_s64(pg, idx_ptr);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, sparse_vec_d, indices);
        
        dot_acc = svmla_f64_z(pg, dot_acc, dense_vec, gathered);
        
        dense_vec_d += vl_d;
        idx_ptr += vl_d;
    }
    
    double dot_sum = svaddv_f64(pg, dot_acc);
    return dot_sum;
}

static void sve_gather_d_vec_idx_fmla_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    last_dot_result = sve_gather_d_vec_idx_fmla_single_reg_core(a, b, c, size, scalar);
}

static double verify_dot_product_ref(void *a, void *c, uint64_t test_size) {
    double *dense_vec = (double *)a;
    double *sparse_vec = (double *)c;
    uint64_t vl_d = svcntd();
    uint64_t iterations = num_nonzero / vl_d;
    
    double ref_sum = 0.0;
    int32_t *idx_ptr = gather_indices;
    
    for (uint64_t i = 0; i < iterations; i++) {
        for (uint64_t j = 0; j < vl_d; j++) {
            uint64_t idx = (uint64_t)idx_ptr[j];
            ref_sum += dense_vec[j] * sparse_vec[idx];
        }
        dense_vec += vl_d;
        idx_ptr += vl_d;
    }
    
    return ref_sum;
}

typedef struct {
    const char *name;
    void (*func)(void *a, void *b, void *c, uint64_t size, double scalar);
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
    printf("  -m, --index-mode <N>    Index generation mode (default: 0)\n");
    printf("                           0: Random, 1: Uniform, 2: RandomUniqueSorted\n");
    printf("  -M, --modulo <N>        Index modulo for RandomUniqueSorted mode (default: sqrt(max_idx))\n");
    printf("                           Limits index range to [0, modulo-1]\n");
    printf("  -r, --random-seed <N>   Random seed for index generation (default: 42)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -p, --print-all         Print all ranks' results (MPI only)\n");
    printf("  -o, --output-indices <file>  Output gather indices to file\n");
    printf("\nExamples:\n");
    printf("  %s -n 1000000 -s 0.01              1M non-zero elements, 1%% sparsity (100MB buffer)\n", prog_name);
    printf("  %s -n 500000 -s 0.02               500K non-zero elements, 2%% sparsity (25MB buffer)\n", prog_name);
    printf("  %s -n 100000 -s 0.001 -m 1         100K non-zero elements, 0.1%% sparsity, uniform indices\n", prog_name);
    printf("  %s -n 100000 -s 0.5 -m 2 -M 1000   100K non-zero, 50%% sparsity, RandomUniqueSorted\n", prog_name);
}

static double run_test(test_item_t *test, void *a, void *b, void *c
#ifdef USE_MPI
    , MPI_Comm comm
#endif
) {
    struct timespec start, end;
    double scalar = 2.0;
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    for (int i = 0; i < warmup_iter; i++) {
        test->func(a, b, c, buffer_size, scalar);
    }
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < test_iter; i++) {
        test->func(a, b, c, buffer_size, scalar);
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
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--index-mode") == 0) {
            if (i + 1 < argc) {
                index_mode = atoi(argv[++i]);
                if (index_mode < 0) index_mode = 0;
                if (index_mode > 2) index_mode = 2;
            }
            continue;
        }
        if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--modulo") == 0) {
            if (i + 1 < argc) {
                index_modulo = (uint64_t)atol(argv[++i]);
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
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random-seed") == 0) {
            if (i + 1 < argc) {
                random_seed = (unsigned int)atoi(argv[++i]);
            }
            continue;
        }
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print-all") == 0) {
            print_all_ranks = 1;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-indices") == 0) {
            if (i + 1 < argc) {
                output_indices_file = argv[++i];
            }
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
    MPI_Bcast(&index_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&index_modulo, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
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
    
    buffer_size = (buffer_size / 1024) * 1024;
    if (buffer_size < 1024) buffer_size = 1024;
    
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
        printf("Random Seed: %u\n", random_seed);
        printf("\n");
    }
    
    void *a = NULL, *b = NULL, *c = NULL;
    
    if (posix_memalign(&a, 64, buffer_size) != 0 ||
        posix_memalign(&b, 64, buffer_size) != 0 ||
        posix_memalign(&c, 64, buffer_size) != 0 ||
        posix_memalign((void**)&gather_indices, 64, num_nonzero * sizeof(int32_t)) != 0) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate aligned memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate aligned memory\n");
#endif
        return 1;
    }
    
    double *da = (double *)a, *db = (double *)b, *dc = (double *)c;
    uint64_t elem_count = buffer_size / sizeof(double);
    for (uint64_t i = 0; i < elem_count; i++) {
        da[i] = 1.0;
        db[i] = 2.0;
        dc[i] = 3.0;
    }
    
    srand(random_seed);
    uint64_t max_element_idx_64 = buffer_size / sizeof(int64_t) - 1;
    uint64_t max_idx = (max_element_idx_64 < INT32_MAX) ? max_element_idx_64 : INT32_MAX;
    
    if (index_mode == 2 && index_modulo == 0) {
        index_modulo = (uint64_t)sqrt((double)max_idx);
    }
    if (index_modulo == 0) index_modulo = max_idx + 1;
    if (index_modulo > max_idx + 1) index_modulo = max_idx + 1;
    
    uint64_t min_idx = max_idx, max_found = 0;
    uint64_t coverage_buckets = (max_idx / 64) + 2;
    uint64_t *coverage = (uint64_t *)calloc(coverage_buckets, sizeof(uint64_t));
    
    const char *mode_names[] = {"Random", "Uniform", "RandomUniqueSorted"};
    uint64_t actual_index_count = num_nonzero;
    
    if (index_mode == 0) {
        for (uint64_t i = 0; i < num_nonzero; i++) {
            uint64_t idx = ((uint64_t)rand() << 32 | rand()) % (max_idx + 1);
            gather_indices[i] = (int32_t)idx;
            update_index_stats(idx, &min_idx, &max_found, coverage);
        }
    } else if (index_mode == 1) {
        uint64_t stride = (max_idx + 1) / num_nonzero;
        if (stride == 0) stride = 1;
        for (uint64_t i = 0; i < num_nonzero; i++) {
            uint64_t idx = i * stride;
            if (idx > max_idx) idx = max_idx;
            gather_indices[i] = (int32_t)idx;
            update_index_stats(idx, &min_idx, &max_found, coverage);
        }
    } else {
        uint64_t max_unique = (max_idx + 1 < num_nonzero) ? max_idx + 1 : num_nonzero;
        uint64_t *unique_indices = (uint64_t *)malloc(max_unique * sizeof(uint64_t));
        uint64_t unique_count = 0;
        int attempts = 0;
        int max_attempts = num_nonzero * 20;
        
        while (unique_count < max_unique && attempts < max_attempts) {
            uint64_t idx = ((uint64_t)rand() << 32 | rand()) % (max_idx + 1);
            uint64_t bucket = idx / 64;
            uint64_t bit = idx % 64;
            if (!(coverage[bucket] & (1ULL << bit))) {
                coverage[bucket] |= (1ULL << bit);
                unique_indices[unique_count++] = idx;
                if (idx < min_idx) min_idx = idx;
                if (idx > max_found) max_found = idx;
            }
            attempts++;
        }
        
        for (uint64_t i = 0; i <= max_idx && unique_count < max_unique; i++) {
            uint64_t bucket = i / 64;
            uint64_t bit = i % 64;
            if (!(coverage[bucket] & (1ULL << bit))) {
                coverage[bucket] |= (1ULL << bit);
                unique_indices[unique_count++] = i;
                if (i < min_idx) min_idx = i;
                if (i > max_found) max_found = i;
            }
        }
        
        qsort(unique_indices, unique_count, sizeof(uint64_t), compare_uint64);
        actual_index_count = unique_count;
        for (uint64_t i = 0; i < unique_count; i++) {
            uint64_t idx = unique_indices[i];
            if (index_modulo < max_idx + 1) {
                idx = idx % index_modulo;
            }
            gather_indices[i] = (int32_t)idx;
        }
        free(unique_indices);
    }
    
    uint64_t covered = 0;
    for (uint64_t i = 0; i < coverage_buckets - 1; i++) {
        covered += __builtin_popcountll(coverage[i]);
    }
    free(coverage);
    
    uint64_t modulo_covered = 0;
    if (index_mode == 2 && index_modulo < max_idx + 1) {
        uint64_t *modulo_coverage = (uint64_t *)calloc((index_modulo / 64) + 2, sizeof(uint64_t));
        for (uint64_t i = 0; i < actual_index_count; i++) {
            uint64_t idx = gather_indices[i];
            modulo_coverage[idx / 64] |= (1ULL << (idx % 64));
        }
        for (uint64_t i = 0; i < (index_modulo / 64) + 1; i++) {
            modulo_covered += __builtin_popcountll(modulo_coverage[i]);
        }
        free(modulo_coverage);
    }
    
    if (rank == 0) {
        printf("Index Mode: %s\n", mode_names[index_mode]);
        if (index_mode == 2 && index_modulo < max_idx + 1) {
            printf("Index Modulo: %lu (sqrt=%.2f)\n", index_modulo, sqrt((double)max_idx));
            printf("Modulo Coverage: %lu / %lu (%.2f%%) of [0, %lu]\n", 
                   modulo_covered, index_modulo, (double)modulo_covered / index_modulo * 100.0, index_modulo - 1);
        }
        printf("Max Index: %lu (buffer elements: %lu)\n", max_idx, max_element_idx_64);
        printf("Generated Range (pre-modulo): [%lu, %lu]\n", min_idx, max_found);
        printf("Unique Indices (pre-modulo): %lu / %lu (%.2f%%)\n", covered, num_nonzero, 
               (double)covered / num_nonzero * 100.0);
        printf("Coverage: %.4f%% of buffer\n\n", 
               (double)covered / (max_idx + 1) * 100.0);
        
        if (output_indices_file) {
            FILE *fp = fopen(output_indices_file, "w");
            if (fp) {
                fprintf(fp, "# Gather Indices Output (Single-Reg)\n");
                fprintf(fp, "# Index Mode: %s\n", mode_names[index_mode]);
                if (index_mode == 2 && index_modulo < max_idx + 1) {
                    fprintf(fp, "# Index Modulo: %lu (applied after sort)\n", index_modulo);
                    fprintf(fp, "# Modulo Coverage: [0, %lu]\n", index_modulo - 1);
                }
                fprintf(fp, "# Random Seed: %u\n", random_seed);
                fprintf(fp, "# Non-Zero Elements: %lu\n", num_nonzero);
                fprintf(fp, "# Sparsity: %.4f\n", sparsity);
                fprintf(fp, "# Max Index: %lu\n", max_idx);
                fprintf(fp, "# Generated Range (pre-modulo): [%lu, %lu]\n", min_idx, max_found);
                fprintf(fp, "# Unique Indices (pre-modulo): %lu\n", covered);
                if (index_mode == 2 && index_modulo < max_idx + 1) {
                    fprintf(fp, "# Modulo Coverage: %lu / %lu (%.2f%%)\n", 
                            modulo_covered, index_modulo, (double)modulo_covered / index_modulo * 100.0);
                }
                fprintf(fp, "# Format: Index | Double_Offset(bytes)\n");
                fprintf(fp, "#\n");
                for (uint64_t i = 0; i < num_nonzero; i++) {
                    int32_t idx = gather_indices[i];
                    fprintf(fp, "%10d %15lu\n", idx, (uint64_t)idx * 8);
                }
                fclose(fp);
                printf("Indices written to: %s\n", output_indices_file);
            } else {
                fprintf(stderr, "Failed to open output file: %s\n", output_indices_file);
            }
        }
    }
    
    if (rank == 0) {
#ifdef USE_MPI
        printf("%-38s %10s %10s %10s %12s %10s\n", 
               "Test", "GB/s", "Time(ms)", "Data(MB)", "Verify", "Total(GB/s)");
#else
        printf("%-38s %10s %10s %10s %12s\n", 
               "Test", "GB/s", "Time(ms)", "Data(MB)", "Verify");
#endif
        printf("================================================================================\n");
    }
    
    for (int i = 0; i < test_count; i++) {
        test_item_t *test = &test_registry[i];
        uint64_t bytes_per_iter = num_nonzero * sizeof(double) * 2;
        
#ifdef USE_MPI
        double time_sec = run_test(test, a, b, c, MPI_COMM_WORLD);
#else
        double time_sec = run_test(test, a, b, c);
#endif
        double bandwidth = get_bandwidth(bytes_per_iter, time_sec);
        
#ifdef USE_MPI
        double total_bw = 0.0;
        MPI_Reduce(&bandwidth, &total_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
        
        int verify_result = -1;
        
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        
        double ref_result = verify_dot_product_ref(a, c, buffer_size);
        double rel_error = fabs(last_dot_result - ref_result) / fabs(ref_result);
        
        if (rel_error < 1e-10) {
            verify_result = 0;
        } else {
            verify_result = 1;
            if (rank == 0) {
                printf("  [Verify] SVE=%.15e REF=%.15e Error=%.15e\n", 
                       last_dot_result, ref_result, rel_error);
            }
        }
        
        if (rank == 0 || print_all_ranks) {
#ifdef USE_MPI
            if (print_all_ranks) {
                printf("[Rank %d] %-38s %10.2f %10.3f %10.0f",
                       rank, test->name, bandwidth, time_sec * 1000,
                       (double)bytes_per_iter / (1024 * 1024));
            } else {
                printf("%-38s %10.2f %10.3f %10.0f",
                       test->name, bandwidth, time_sec * 1000,
                       (double)bytes_per_iter / (1024 * 1024));
            }
#else
            printf("%-38s %10.2f %10.3f %10.0f",
                   test->name, bandwidth, time_sec * 1000,
                   (double)bytes_per_iter / (1024 * 1024));
#endif
            
            if (verify_result > 0) {
                printf("  FAIL(%d)", verify_result);
            } else if (verify_result == 0) {
                printf("  PASS");
            } else {
                printf("  NO_CHECK");
            }
            
#ifdef USE_MPI
            if (!print_all_ranks) {
                printf(" %10.2f", total_bw);
            }
#endif
            printf("\n");
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
    free(c);
    free(gather_indices);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}