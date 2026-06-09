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
static uint64_t index_pool_size = 0;
static int32_t *gather_indices = NULL;
static unsigned int random_seed = 42;
static int print_all_ranks = 0;

#pragma GCC push_options
#pragma GCC optimize ("O3")

static void sve_gather_d_idx_only_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * sizeof(int32_t);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x21, %[idx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x21, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1sw z8.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z8.d, lsl 3]\n"
        "add x21, x21, x19\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        :
        : [sd] "r" (src_d), [idx] "r" (idx_base), [inc] "r" (idx_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x21", "p0",
          "z0", "z8", "memory"
    );
}

static void sve_gather_d_vec_idx_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * sizeof(int32_t);
    uint64_t dst_inc = vl_d * sizeof(double);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[incd]\n"
        "mov x21, %[idx]\n"
        "mov x22, %[vx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x21, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1d z4.d, p0/z, [x22, #0, MUL VL]\n"
        "ld1sw z8.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z8.d, lsl 3]\n"
        "add x21, x21, x19\n"
        "add x22, x22, x20\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        :
        : [sd] "r" (src_d), [vx] "r" (vec_x_d), [idx] "r" (idx_base), [inc] "r" (idx_inc), [incd] "r" (dst_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "x21", "x22", "p0",
          "z0", "z4", "z8", "memory"
    );
}

static void sve_gather_d_vec_idx_fmla_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * sizeof(int32_t);
    uint64_t dst_inc = vl_d * sizeof(double);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[incd]\n"
        "mov x21, %[idx]\n"
        "mov x22, %[vx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x21, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1d z4.d, p0/z, [x22, #0, MUL VL]\n"
        "ld1sw z8.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z8.d, lsl 3]\n"
        "fmla z4.d, p0/m, z4.d, z0.d\n"
        "add x21, x21, x19\n"
        "add x22, x22, x20\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        :
        : [sd] "r" (src_d), [vx] "r" (vec_x_d), [idx] "r" (idx_base), [inc] "r" (idx_inc), [incd] "r" (dst_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "x21", "x22", "p0",
          "z0", "z4", "z8", "memory"
    );
}

static void sve_gather_d_idx_store_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * sizeof(int32_t);
    uint64_t dst_inc = vl_d * sizeof(double);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[incd]\n"
        "mov x21, %[idx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x21, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1sw z8.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z8.d, lsl 3]\n"
        "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
        "add x21, x21, x19\n"
        "add %[d], %[d], x20\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        : [d] "+r" (dst)
        : [sd] "r" (src_d), [idx] "r" (idx_base), [inc] "r" (idx_inc), [incd] "r" (dst_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "x21", "p0",
          "z0", "z8", "memory"
    );
}

static void sve_gather_d_vec_idx_store_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * sizeof(int32_t);
    uint64_t dst_inc = vl_d * sizeof(double);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[incd]\n"
        "mov x21, %[idx]\n"
        "mov x22, %[vx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x21, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1d z4.d, p0/z, [x22, #0, MUL VL]\n"
        "ld1sw z8.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z8.d, lsl 3]\n"
        "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
        "add x21, x21, x19\n"
        "add %[d], %[d], x20\n"
        "add x22, x22, x20\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        : [d] "+r" (dst)
        : [sd] "r" (src_d), [vx] "r" (vec_x_d), [idx] "r" (idx_base), [inc] "r" (idx_inc), [incd] "r" (dst_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "x21", "x22", "p0",
          "z0", "z4", "z8", "memory"
    );
}

static void sve_gather_d_vec_idx_fmla_store_single_reg(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    double *vec_x_d = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / vl_d;
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * sizeof(int32_t);
    uint64_t dst_inc = vl_d * sizeof(double);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[incd]\n"
        "mov x21, %[idx]\n"
        "mov x22, %[vx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x21, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1d z4.d, p0/z, [x22, #0, MUL VL]\n"
        "ld1sw z8.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z8.d, lsl 3]\n"
        "fmla z4.d, p0/m, z4.d, z0.d\n"
        "st1d z4.d, p0, [%[d], #0, MUL VL]\n"
        "add x21, x21, x19\n"
        "add %[d], %[d], x20\n"
        "add x22, x22, x20\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        : [d] "+r" (dst)
        : [sd] "r" (src_d), [vx] "r" (vec_x_d), [idx] "r" (idx_base), [inc] "r" (idx_inc), [incd] "r" (dst_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "x21", "x22", "p0",
          "z0", "z4", "z8", "memory"
    );
}

#pragma GCC pop_options

static inline uint64_t calc_idx_pos(uint64_t i, uint64_t chunk, uint64_t pool_iters) {
    return (i / chunk % pool_iters) * chunk + i % chunk;
}

static int verify_gather_single_reg(void *dst_ptr, void *src_ptr) {
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
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
    uint64_t vl_d = svcntb() / sizeof(int64_t);
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
} test_item_t;

static test_item_t test_registry[] = {
    {"SVE Gather D IdxOnly (Single-Reg)",        sve_gather_d_idx_only_single_reg},
    {"SVE Gather D Vec+Idx (Single-Reg)",        sve_gather_d_vec_idx_single_reg},
    {"SVE Gather D Vec+Idx+FMLA (Single-Reg)",   sve_gather_d_vec_idx_fmla_single_reg},
    {"SVE Gather D Idx+Store (Single-Reg)",      sve_gather_d_idx_store_single_reg},
    {"SVE Gather D Vec+Idx+Store (Single-Reg)",  sve_gather_d_vec_idx_store_single_reg},
    {"SVE Gather D Vec+Idx+FMLA+Store (Single-Reg)", sve_gather_d_vec_idx_fmla_store_single_reg},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
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
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--buffer-size") == 0) {
            if (i + 1 < argc) {
                buffer_size = (uint64_t)atoi(argv[++i]) * 1024 * 1024;
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
    }
    
#ifdef USE_MPI
    MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    uint64_t vl = svcntb();
    uint64_t vl_d = vl / sizeof(int64_t);
    
    buffer_size = (buffer_size / 1024) * 1024;
    if (buffer_size < 1024) buffer_size = 1024;
    
    uint64_t max_element_idx = buffer_size / sizeof(double) - 1;
    uint64_t max_idx = (max_element_idx < INT32_MAX) ? max_element_idx : INT32_MAX;
    
    index_pool_size = buffer_size / sizeof(double);
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
    for (uint64_t i = 0; i < index_pool_size; i++) {
        uint64_t idx = ((uint64_t)rand() << 32 | rand()) % (max_idx + 1);
        gather_indices[i] = (int32_t)idx;
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
        if (test->func == sve_gather_d_idx_store_single_reg || 
            test->func == sve_gather_d_vec_idx_store_single_reg) {
            verify_result = verify_gather_single_reg(a, c);
        } else if (test->func == sve_gather_d_vec_idx_fmla_store_single_reg) {
            verify_result = verify_gather_d_fmla_single_reg(a, c, b);
        }
        
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