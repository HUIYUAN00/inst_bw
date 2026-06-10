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
static uint64_t buffer_size = 128 * 1024 * 1024;
static double sparsity = 1.0;
static int index_mode = 0;
static int print_all_ranks = 0;
static uint64_t index_pool_size = 0;
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

#pragma GCC push_options
#pragma GCC optimize ("O3")

static void sve_gather_d_idx_only_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dummy_dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntd();
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    svbool_t pg = svptrue_b64();
    uint64_t pool_counter = 0;
    svfloat64_t acc = svdup_f64(0.0);
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (pool_counter == 0) {
            idx_base = gather_indices;
            pool_counter = idx_pool_iters;
        }
        
        svint64_t indices = svld1sw_s64(pg, idx_base);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, src_d, indices);
        acc = svadd_f64_z(pg, acc, gathered);
        
        idx_base += vl_d;
        pool_counter--;
    }
    
    svst1_f64(pg, dummy_dst, acc);
}

static void sve_gather_d_vec_idx_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *vec_x_d = (double *)b;
    double *dummy_dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntd();
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    svbool_t pg = svptrue_b64();
    uint64_t pool_counter = 0;
    svfloat64_t acc = svdup_f64(0.0);
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (pool_counter == 0) {
            idx_base = gather_indices;
            pool_counter = idx_pool_iters;
        }
        
        svfloat64_t vec_x = svld1_f64(pg, vec_x_d);
        svint64_t indices = svld1sw_s64(pg, idx_base);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, src_d, indices);
        acc = svadd_f64_z(pg, acc, svadd_f64_z(pg, vec_x, gathered));
        
        idx_base += vl_d;
        vec_x_d += vl_d;
        pool_counter--;
    }
    
    svst1_f64(pg, dummy_dst, acc);
}

static double sve_gather_d_vec_idx_fmla_single_reg_core(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntd();
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    svbool_t pg = svptrue_b64();
    uint64_t pool_counter = 0;
    double dot_sum = 0.0;
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (pool_counter == 0) {
            idx_base = gather_indices;
            pool_counter = idx_pool_iters;
        }
        
        svfloat64_t vec_x = svld1_f64(pg, vec_x_d);
        svint64_t indices = svld1sw_s64(pg, idx_base);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, src_d, indices);
        vec_x = svmla_f64_z(pg, vec_x, vec_x, gathered);
        dot_sum += svaddv_f64(pg, vec_x);
        
        idx_base += vl_d;
        vec_x_d += vl_d;
        pool_counter--;
    }
    
    return dot_sum;
}

static void sve_gather_d_vec_idx_fmla_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    last_dot_result = sve_gather_d_vec_idx_fmla_single_reg_core(a, b, c, size, scalar);
}

static void sve_gather_d_idx_store_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntd();
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    svbool_t pg = svptrue_b64();
    uint64_t pool_counter = 0;
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (pool_counter == 0) {
            idx_base = gather_indices;
            pool_counter = idx_pool_iters;
        }
        
        svint64_t indices = svld1sw_s64(pg, idx_base);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, src_d, indices);
        svst1_f64(pg, dst, gathered);
        
        idx_base += vl_d;
        dst += vl_d;
        pool_counter--;
    }
}

static void sve_gather_d_vec_idx_store_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntd();
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    svbool_t pg = svptrue_b64();
    uint64_t pool_counter = 0;
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (pool_counter == 0) {
            idx_base = gather_indices;
            pool_counter = idx_pool_iters;
        }
        
        svfloat64_t vec_x = svld1_f64(pg, vec_x_d);
        svint64_t indices = svld1sw_s64(pg, idx_base);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, src_d, indices);
        svst1_f64(pg, dst, gathered);
        
        idx_base += vl_d;
        dst += vl_d;
        vec_x_d += vl_d;
        pool_counter--;
    }
}

static void sve_gather_d_vec_idx_fmla_store_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntd();
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    svbool_t pg = svptrue_b64();
    uint64_t pool_counter = 0;
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (pool_counter == 0) {
            idx_base = gather_indices;
            pool_counter = idx_pool_iters;
        }
        
        svfloat64_t vec_x = svld1_f64(pg, vec_x_d);
        svint64_t indices = svld1sw_s64(pg, idx_base);
        svfloat64_t gathered = svld1_gather_s64index_f64(pg, src_d, indices);
        vec_x = svmla_f64_z(pg, vec_x, vec_x, gathered);
        svst1_f64(pg, dst, vec_x);
        
        idx_base += vl_d;
        dst += vl_d;
        vec_x_d += vl_d;
        pool_counter--;
    }
}

#pragma GCC pop_options

static inline uint64_t calc_idx_pos(uint64_t i, uint64_t chunk, uint64_t pool_iters) {
    return (i / chunk % pool_iters) * chunk + i % chunk;
}

static int verify_gather_single_reg(void *dst_ptr, void *src_ptr) {
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl_d = svcntd();
    uint64_t chunk = vl_d;
    uint64_t pool_iters = index_pool_size / chunk;
    if (pool_iters < 1) pool_iters = 1;
    uint64_t total = buffer_size / sizeof(double);
    
    double *src = (double *)src_ptr;
    double *dst = (double *)dst_ptr;
    
    for (uint64_t i = 0; i < total && errors < 5; i++) {
        uint64_t idx_pos = calc_idx_pos(i, chunk, pool_iters);
        if (idx_pos >= index_pool_size) continue;
        int32_t elem_idx = indices[idx_pos];
        
        if (src[elem_idx] != dst[i]) {
            if (errors == 0) fprintf(stderr, "Gather Single-Reg verify FAILED:\n");
            fprintf(stderr, "  dst[%lu]: expected %.1f (src[%d]), got %.1f\n", 
                    i, src[elem_idx], elem_idx, dst[i]);
            errors++;
        }
    }
    return errors;
}

static int verify_gather_d_fmla_single_reg(void *dst_ptr, void *src_ptr, void *vec_x_ptr) {
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl_d = svcntd();
    uint64_t chunk = vl_d;
    uint64_t pool_iters = index_pool_size / chunk;
    if (pool_iters < 1) pool_iters = 1;
    uint64_t total = buffer_size / sizeof(double);
    
    double *src = (double *)src_ptr;
    double *dst = (double *)dst_ptr;
    double *vec_x = (double *)vec_x_ptr;
    
    for (uint64_t i = 0; i < total && errors < 5; i++) {
        uint64_t idx_pos = calc_idx_pos(i, chunk, pool_iters);
        if (idx_pos >= index_pool_size) continue;
        int32_t elem_idx = indices[idx_pos];
        
        double expected = vec_x[i] + vec_x[i] * src[elem_idx];
        if (dst[i] != expected) {
            if (errors == 0) fprintf(stderr, "FMLA Single-Reg verify FAILED:\n");
            fprintf(stderr, "  dst[%lu]: expected %.1f (vec_x[%lu=%.1f]+vec_x*src[%d=%.1f]), got %.1f\n",
                    i, expected, i, vec_x[i], elem_idx, src[elem_idx], dst[i]);
            errors++;
        }
    }
    return errors;
}

typedef struct {
    const char *name;
    void (*func)(void *a, void *b, void *c, uint64_t size, double scalar);
    double (*func_dot)(void *a, void *b, void *c, uint64_t size, double scalar);
} test_item_t;

static test_item_t test_registry[] = {
    {"SVE Gather D Vec+Idx+FMLA (Single-Reg)",   sve_gather_d_vec_idx_fmla_single_reg},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -b, --buffer-size <MB>  Buffer size in MB (default: 128)\n");
    printf("  -s, --sparsity <ratio>  Sparsity ratio 0.0-1.0 (default: 1.0)\n");
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
    printf("  %s                               Run with default settings\n", prog_name);
    printf("  %s -b 64 -s 0.02                 64MB buffer, 2%% sparsity\n", prog_name);
    printf("  %s -s 1.0 -m 1                   Full range, uniform indices\n", prog_name);
    printf("  %s -m 2 -M 1000                  RandomUniqueSorted mode with modulo=1000\n", prog_name);
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
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--buffer-size") == 0) {
            if (i + 1 < argc) {
                buffer_size = (uint64_t)atoi(argv[++i]) * 1024 * 1024;
            }
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
    
#ifdef USE_MPI
    MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
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
    
    buffer_size = (buffer_size / 1024) * 1024;
    if (buffer_size < 1024) buffer_size = 1024;
    
    index_pool_size = (uint64_t)(sparsity * (buffer_size / sizeof(int64_t)));
    uint64_t min_indices = vl_d * 2;
    if (index_pool_size < min_indices) index_pool_size = min_indices;
    
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
        printf("Buffer Size: %lu MB per array\n", buffer_size / (1024 * 1024));
        printf("Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("Index Pool Size: %lu elements\n", index_pool_size);
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
        posix_memalign((void**)&gather_indices, 64, index_pool_size * sizeof(int32_t)) != 0) {
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
    
    if (index_mode == 0) {
        for (uint64_t i = 0; i < index_pool_size; i++) {
            uint64_t idx = ((uint64_t)rand() << 32 | rand()) % (max_idx + 1);
            gather_indices[i] = (int32_t)idx;
            update_index_stats(idx, &min_idx, &max_found, coverage);
        }
    } else if (index_mode == 1) {
        uint64_t stride = (max_idx + 1) / index_pool_size;
        if (stride == 0) stride = 1;
        for (uint64_t i = 0; i < index_pool_size; i++) {
            uint64_t idx = i * stride;
            if (idx > max_idx) idx = max_idx;
            gather_indices[i] = (int32_t)idx;
            update_index_stats(idx, &min_idx, &max_found, coverage);
        }
    } else {
        uint64_t max_unique = (max_idx + 1 < index_pool_size) ? max_idx + 1 : index_pool_size;
        uint64_t *unique_indices = (uint64_t *)malloc(max_unique * sizeof(uint64_t));
        uint64_t unique_count = 0;
        int attempts = 0;
        int max_attempts = index_pool_size * 20;
        
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
        index_pool_size = unique_count;
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
        for (uint64_t i = 0; i < index_pool_size; i++) {
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
        printf("Unique Indices (pre-modulo): %lu / %lu (%.2f%%)\n", covered, index_pool_size, 
               (double)covered / index_pool_size * 100.0);
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
                fprintf(fp, "# Sparsity: %.4f\n", sparsity);
                fprintf(fp, "# Index Pool Size: %lu\n", index_pool_size);
                fprintf(fp, "# Max Index: %lu\n", max_idx);
                fprintf(fp, "# Generated Range (pre-modulo): [%lu, %lu]\n", min_idx, max_found);
                fprintf(fp, "# Unique Indices (pre-modulo): %lu\n", covered);
                if (index_mode == 2 && index_modulo < max_idx + 1) {
                    fprintf(fp, "# Modulo Coverage: %lu / %lu (%.2f%%)\n", 
                            modulo_covered, index_modulo, (double)modulo_covered / index_modulo * 100.0);
                }
                fprintf(fp, "# Format: Index | Double_Offset(bytes)\n");
                fprintf(fp, "#\n");
                for (uint64_t i = 0; i < index_pool_size; i++) {
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
        uint64_t bytes_per_iter = buffer_size * 2;
        
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