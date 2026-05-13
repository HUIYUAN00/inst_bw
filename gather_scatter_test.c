#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_sve.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static int warmup_iter = 5;
static int test_iter = 10;
static uint64_t buffer_size = 128 * 1024 * 1024;
static double sparsity = 0.01;
static int index_mode = 0;
static int print_all_ranks = 0;
static uint64_t index_pool_size = 0;
static int32_t *gather_indices = NULL;

typedef struct {
    const char *name;
    const char *category;
    void (*func)(void *a, void *b, void *c, uint64_t size, double scalar);
} test_item_t;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

//=== SVE_GATHER_TESTS

static void sve_gather_ld1w_ld1w(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *src = (float *)b;
    float *dst = (float *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl = svcntb() / sizeof(int32_t);
    uint64_t chunk_bytes = vl * 4 * sizeof(int32_t);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / (vl * 4);
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[idx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x20, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.s\n"
        "ld1w z8.s, p0/z, [x20, #0, MUL VL]\n"
        "ld1w z9.s, p0/z, [x20, #1, MUL VL]\n"
        "ld1w z10.s, p0/z, [x20, #2, MUL VL]\n"
        "ld1w z11.s, p0/z, [x20, #3, MUL VL]\n"
        "ld1w z0.s, p0/z, [%[s], z8.s, sxtw 2]\n"
        "ld1w z1.s, p0/z, [%[s], z9.s, sxtw 2]\n"
        "ld1w z2.s, p0/z, [%[s], z10.s, sxtw 2]\n"
        "ld1w z3.s, p0/z, [%[s], z11.s, sxtw 2]\n"
        "st1w z0.s, p0, [%[d], #0, MUL VL]\n"
        "st1w z1.s, p0, [%[d], #1, MUL VL]\n"
        "st1w z2.s, p0, [%[d], #2, MUL VL]\n"
        "st1w z3.s, p0, [%[d], #3, MUL VL]\n"
        "add x20, x20, x19\n"
        "add %[d], %[d], x19\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        : [d] "+r" (dst)
        : [s] "r" (src), [idx] "r" (idx_base), [inc] "r" (chunk_bytes),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "p0",
          "z0", "z1", "z2", "z3", "z8", "z9", "z10", "z11", "memory"
    );
}

static void sve_gather_ld1sw_ld1d(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_bytes = svcntb();
    uint64_t vl_d = vl_bytes / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * 4 * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / (vl_d * 4);
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * 4 * sizeof(int32_t);
    uint64_t dst_inc = vl_d * 4 * sizeof(double);
    
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
        "ld1sw z4.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1sw z5.d, p0/z, [x21, #1, MUL VL]\n"
        "ld1sw z6.d, p0/z, [x21, #2, MUL VL]\n"
        "ld1sw z7.d, p0/z, [x21, #3, MUL VL]\n"
        "ld1d z0.d, p0/z, [%[sd], z4.d, lsl 3]\n"
        "ld1d z1.d, p0/z, [%[sd], z5.d, lsl 3]\n"
        "ld1d z2.d, p0/z, [%[sd], z6.d, lsl 3]\n"
        "ld1d z3.d, p0/z, [%[sd], z7.d, lsl 3]\n"
        "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
        "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
        "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
        "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
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
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
    );
}

//=== SVE_SCATTER_TESTS

static void sve_scatter_st1w(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *src = (float *)a;
    float *dst = (float *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl = svcntb() / sizeof(int32_t);
    uint64_t chunk_bytes = vl * 4 * sizeof(float);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / (vl * 4);
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[idx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x20, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.s\n"
        "ld1w z0.s, p0/z, [%[s], #0, MUL VL]\n"
        "ld1w z1.s, p0/z, [%[s], #1, MUL VL]\n"
        "ld1w z2.s, p0/z, [%[s], #2, MUL VL]\n"
        "ld1w z3.s, p0/z, [%[s], #3, MUL VL]\n"
        "ld1w z8.s, p0/z, [x20, #0, MUL VL]\n"
        "ld1w z9.s, p0/z, [x20, #1, MUL VL]\n"
        "ld1w z10.s, p0/z, [x20, #2, MUL VL]\n"
        "ld1w z11.s, p0/z, [x20, #3, MUL VL]\n"
        "st1w z0.s, p0, [%[d], z8.s, sxtw 2]\n"
        "st1w z1.s, p0, [%[d], z9.s, sxtw 2]\n"
        "st1w z2.s, p0, [%[d], z10.s, sxtw 2]\n"
        "st1w z3.s, p0, [%[d], z11.s, sxtw 2]\n"
        "add x20, x20, x19\n"
        "add %[s], %[s], x19\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        : [s] "+r" (src)
        : [d] "r" (dst), [idx] "r" (idx_base), [inc] "r" (chunk_bytes),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "p0",
          "z0", "z1", "z2", "z3", "z8", "z9", "z10", "z11", "memory"
    );
}

static void sve_scatter_st1d(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src = (double *)a;
    double *dst = (double *)b;
    int32_t *idx_base = gather_indices;
    uint64_t vl_bytes = svcntb();
    uint64_t vl_d = vl_bytes / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * 4 * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / (vl_d * 4);
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * 4 * sizeof(int32_t);
    uint64_t src_inc = vl_d * 4 * sizeof(double);
    
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
        "ld1d z0.d, p0/z, [%[s], #0, MUL VL]\n"
        "ld1d z1.d, p0/z, [%[s], #1, MUL VL]\n"
        "ld1d z2.d, p0/z, [%[s], #2, MUL VL]\n"
        "ld1d z3.d, p0/z, [%[s], #3, MUL VL]\n"
        "ld1sw z12.d, p0/z, [x21, #0, MUL VL]\n"
        "ld1sw z13.d, p0/z, [x21, #1, MUL VL]\n"
        "ld1sw z14.d, p0/z, [x21, #2, MUL VL]\n"
        "ld1sw z15.d, p0/z, [x21, #3, MUL VL]\n"
        "st1d z0.d, p0, [%[d], z12.d, lsl 3]\n"
        "st1d z1.d, p0, [%[d], z13.d, lsl 3]\n"
        "st1d z2.d, p0, [%[d], z14.d, lsl 3]\n"
        "st1d z3.d, p0, [%[d], z15.d, lsl 3]\n"
        "add x21, x21, x19\n"
        "add %[s], %[s], x20\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        : [s] "+r" (src)
        : [d] "r" (dst), [idx] "r" (idx_base), [inc] "r" (idx_inc), [incd] "r" (src_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "x21", "p0",
          "z0", "z1", "z2", "z3", "z12", "z13", "z14", "z15", "memory"
    );
}

//=== SVE_GATHER_SCATTER_TESTS

static void sve_gather_scatter_w(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *src = (float *)b;
    float *dst = (float *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl = svcntb() / sizeof(int32_t);
    uint64_t chunk_bytes = vl * 4 * sizeof(float);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / (vl * 4);
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[idx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x20, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.s\n"
        "ld1w z8.s, p0/z, [x20, #0, MUL VL]\n"
        "ld1w z9.s, p0/z, [x20, #1, MUL VL]\n"
        "ld1w z10.s, p0/z, [x20, #2, MUL VL]\n"
        "ld1w z11.s, p0/z, [x20, #3, MUL VL]\n"
        "ld1w z0.s, p0/z, [%[s], z8.s, sxtw 2]\n"
        "ld1w z1.s, p0/z, [%[s], z9.s, sxtw 2]\n"
        "ld1w z2.s, p0/z, [%[s], z10.s, sxtw 2]\n"
        "ld1w z3.s, p0/z, [%[s], z11.s, sxtw 2]\n"
        "st1w z0.s, p0, [%[d], z8.s, sxtw 2]\n"
        "st1w z1.s, p0, [%[d], z9.s, sxtw 2]\n"
        "st1w z2.s, p0, [%[d], z10.s, sxtw 2]\n"
        "st1w z3.s, p0, [%[d], z11.s, sxtw 2]\n"
        "add x20, x20, x19\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        :
        : [s] "r" (src), [d] "r" (dst), [idx] "r" (idx_base), [inc] "r" (chunk_bytes),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "p0",
          "z0", "z1", "z2", "z3", "z8", "z9", "z10", "z11", "memory"
    );
}

static void sve_gather_scatter_d(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *src = (double *)b;
    double *dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_bytes = svcntb();
    uint64_t vl_d = vl_bytes / sizeof(int64_t);
    uint64_t chunk_bytes = vl_d * 4 * sizeof(double);
    uint64_t iterations = buffer_size / chunk_bytes;
    uint64_t idx_pool_iters = index_pool_size / (vl_d * 4);
    if (idx_pool_iters < 1) idx_pool_iters = 1;
    
    uint64_t idx_inc = vl_d * 4 * sizeof(int32_t);
    
    __asm__ volatile (
        "mov x16, %[iter]\n"
        "mov x17, #0\n"
        "mov x18, %[idx_reset]\n"
        "mov x19, %[inc]\n"
        "mov x20, %[idx]\n"
        "1:\n"
        "cmp x17, #0\n"
        "b.ne 2f\n"
        "mov x20, x18\n"
        "mov x17, %[reset]\n"
        "2:\n"
        "ptrue p0.d\n"
        "ld1sw z4.d, p0/z, [x20, #0, MUL VL]\n"
        "ld1sw z5.d, p0/z, [x20, #1, MUL VL]\n"
        "ld1sw z6.d, p0/z, [x20, #2, MUL VL]\n"
        "ld1sw z7.d, p0/z, [x20, #3, MUL VL]\n"
        "ld1d z8.d, p0/z, [%[s], z4.d, lsl 3]\n"
        "ld1d z9.d, p0/z, [%[s], z5.d, lsl 3]\n"
        "ld1d z10.d, p0/z, [%[s], z6.d, lsl 3]\n"
        "ld1d z11.d, p0/z, [%[s], z7.d, lsl 3]\n"
        "st1d z8.d, p0, [%[d], z4.d, lsl 3]\n"
        "st1d z9.d, p0, [%[d], z5.d, lsl 3]\n"
        "st1d z10.d, p0, [%[d], z6.d, lsl 3]\n"
        "st1d z11.d, p0, [%[d], z7.d, lsl 3]\n"
        "add x20, x20, x19\n"
        "subs x17, x17, #1\n"
        "subs x16, x16, #1\n"
        "b.ne 1b\n"
        :
        : [s] "r" (src), [d] "r" (dst), [idx] "r" (idx_base), [inc] "r" (idx_inc),
          [iter] "r" (iterations), [reset] "r" (idx_pool_iters),
          [idx_reset] "r" (gather_indices)
        : "x16", "x17", "x18", "x19", "x20", "p0",
          "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "memory"
    );
}

//=== End

#pragma GCC pop_options

//=== VERIFY_FUNCTIONS

static inline uint64_t calc_idx_pos(uint64_t i, uint64_t chunk, uint64_t pool_iters) {
    return (i / chunk % pool_iters) * chunk + i % chunk;
}

static int verify_gather(void *dst_ptr, void *src_ptr, int is_double) {
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl = is_double ? svcntb() / sizeof(int64_t) : svcntb() / sizeof(int32_t);
    uint64_t chunk = vl * 4;
    uint64_t pool_iters = index_pool_size / chunk;
    if (pool_iters < 1) pool_iters = 1;
    uint64_t total = buffer_size / (is_double ? sizeof(double) : sizeof(float));
    
    for (uint64_t i = 0; i < total && errors < 5; i++) {
        uint64_t idx_pos = calc_idx_pos(i, chunk, pool_iters);
        if (idx_pos >= index_pool_size) continue;
        int32_t elem_idx = indices[idx_pos];
        
        if (is_double) {
            double *src = (double *)src_ptr;
            double *dst = (double *)dst_ptr;
            if (src[elem_idx] != dst[i]) {
                if (errors == 0) fprintf(stderr, "LD1SW+LD1D Gather verify FAILED:\n");
                fprintf(stderr, "  dst[%lu]: expected %.1f (src[%d]), got %.1f\n", i, src[elem_idx], elem_idx, dst[i]);
                errors++;
            }
        } else {
            float *src = (float *)src_ptr;
            float *dst = (float *)dst_ptr;
            if (src[elem_idx] != dst[i]) {
                if (errors == 0) fprintf(stderr, "LD1W Gather verify FAILED:\n");
                fprintf(stderr, "  dst[%lu]: expected %.1f (src[%d]), got %.1f\n", i, src[elem_idx], elem_idx, dst[i]);
                errors++;
            }
        }
    }
    return errors;
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

static int verify_scatter_common(void *src_ptr, void *dst_ptr, int is_double, int gather_mode, const char *test_name) {
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl = is_double ? svcntb() / sizeof(int64_t) : svcntb() / sizeof(int32_t);
    uint64_t chunk = vl * 4;
    uint64_t pool_iters = index_pool_size / chunk;
    if (pool_iters < 1) pool_iters = 1;
    uint64_t total = buffer_size / (is_double ? sizeof(double) : sizeof(float));
    
    uint64_t *write_count = (uint64_t *)malloc(total * sizeof(uint64_t));
    if (!write_count) { fprintf(stderr, "%s: malloc failed\n", test_name); return -1; }
    void *expected_val = malloc(total * (is_double ? sizeof(double) : sizeof(float)));
    if (!expected_val) { free(write_count); fprintf(stderr, "%s: malloc failed\n", test_name); return -1; }
    memset(write_count, 0, total * sizeof(uint64_t));
    
    for (uint64_t i = 0; i < total; i++) {
        uint64_t idx_pos = calc_idx_pos(i, chunk, pool_iters);
        if (idx_pos >= index_pool_size) continue;
        int32_t elem_idx = indices[idx_pos];
        if (elem_idx >= 0 && elem_idx < total) {
            write_count[elem_idx]++;
            uint64_t src_idx = gather_mode ? elem_idx : i;
            if (is_double) ((double *)expected_val)[elem_idx] = ((double *)src_ptr)[src_idx];
            else ((float *)expected_val)[elem_idx] = ((float *)src_ptr)[src_idx];
        }
    }
    
    int verified = 0;
    for (uint64_t i = 0; i < total && verified < 100; i++) {
        if (write_count[i] == 1) {
            double exp_d, act_d;
            float exp_f, act_f;
            if (is_double) {
                exp_d = ((double *)expected_val)[i];
                act_d = ((double *)dst_ptr)[i];
                if (exp_d != act_d && errors < 5) {
                    if (errors == 0) fprintf(stderr, "%s verify FAILED:\n", test_name);
                    fprintf(stderr, "  dst[%lu]: expected %.1f, got %.1f\n", i, exp_d, act_d);
                    errors++;
                }
            } else {
                exp_f = ((float *)expected_val)[i];
                act_f = ((float *)dst_ptr)[i];
                if (exp_f != act_f && errors < 5) {
                    if (errors == 0) fprintf(stderr, "%s verify FAILED:\n", test_name);
                    fprintf(stderr, "  dst[%lu]: expected %.1f, got %.1f\n", i, exp_f, act_f);
                    errors++;
                }
            }
            verified++;
        }
    }
    
    free(write_count);
    free(expected_val);
    return errors;
}

static int verify_scatter(void *src_ptr, void *dst_ptr, int is_double) {
    const char *name = is_double ? "ST1D Scatter" : "ST1W Scatter";
    return verify_scatter_common(src_ptr, dst_ptr, is_double, 0, name);
}

static int verify_gather_scatter(void *dst_ptr, void *src_ptr, int is_double) {
    const char *name = is_double ? "Gather+Scatter D" : "Gather+Scatter W";
    return verify_scatter_common(src_ptr, dst_ptr, is_double, 1, name);
}

//=== End

//=== TEST_REGISTRY

static test_item_t test_registry[] = {
    {"SVE Gather LD1W (Seq-Store)",     "Gather",        sve_gather_ld1w_ld1w},
    {"SVE Gather LD1SW+LD1D (Seq-Store)", "Gather",        sve_gather_ld1sw_ld1d},
    {"SVE Scatter ST1W (Idx-Store)",    "Scatter",       sve_scatter_st1w},
    {"SVE Scatter ST1D (Idx-Store)",    "Scatter",       sve_scatter_st1d},
    {"SVE Gather+Scatter W (Idx-Store)", "GatherScatter", sve_gather_scatter_w},
    {"SVE Gather+Scatter D (Idx-Store)", "GatherScatter", sve_gather_scatter_d},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options] [test_spec...]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -l, --list              List all available tests\n");
    printf("  -b, --buffer-size <MB>  Buffer size in MB (default: 128)\n");
    printf("  -s, --sparsity <ratio>  Sparsity ratio 0.0-1.0 (default: 0.01)\n");
    printf("  -m, --index-mode <N>    Index generation mode (default: 0)\n");
    printf("                           0: Random, 1: Uniform, 2: Hotspot, 3: RandomUniqueSorted\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -p, --print-all         Print all ranks' results (MPI only)\n");
    printf("\nTest Specification:\n");
    printf("  <index>                 Run test by index (0-based)\n");
    printf("  <name>                  Run test by name (partial match)\n");
    printf("  <category>              Run all tests in a category\n");
    printf("\nCategories: Gather, Scatter, GatherScatter\n");
    printf("\nExamples:\n");
    printf("  %s                               Run all tests (default)\n", prog_name);
    printf("  %s -b 64 -s 0.02                 64MB buffer, 2%% sparsity\n", prog_name);
    printf("  %s -s 1.0 -m 1                   Full range, uniform indices\n", prog_name);
    printf("  %s -s 0.5 -m 2                    50%% sparsity, hotspot pattern\n", prog_name);
    printf("  %s Gather                        Run all Gather tests\n", prog_name);
    printf("  %s 0 2                           Run tests 0 and 2\n", prog_name);
}

static void print_tests(void) {
    printf("Available Tests:\n");
    printf("============================================================\n");
    printf("%-4s %-22s %15s\n", "Idx", "Test Name", "Category");
    printf("============================================================\n");
    for (int i = 0; i < test_count; i++) {
        printf("%-4d %-22s %15s\n", i, test_registry[i].name, test_registry[i].category);
    }
    printf("============================================================\n");
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

//=== End

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
    
    int run_all = 1;
    int num_specs = 0;
    char **specs = NULL;
    
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
        if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            if (rank == 0) {
                print_tests();
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
                if (index_mode > 3) index_mode = 3;
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
        run_all = 0;
        num_specs++;
    }
    
    if (!run_all && num_specs > 0) {
        specs = &argv[argc - num_specs];
    }
    
#ifdef USE_MPI
    MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&index_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    buffer_size = (buffer_size / 1024) * 1024;
    if (buffer_size < 1024) buffer_size = 1024;
    
    index_pool_size = (uint64_t)(sparsity * (buffer_size / sizeof(int64_t)));
    uint64_t vl_test = svcntb() / sizeof(int64_t);
    uint64_t vl_test_32 = svcntb() / sizeof(int32_t);
    uint64_t min_indices_64 = vl_test * 4 * 2;
    uint64_t min_indices_32 = vl_test_32 * 4 * 2;
    uint64_t min_indices = (min_indices_64 > min_indices_32) ? min_indices_64 : min_indices_32;
    if (index_pool_size < min_indices) index_pool_size = min_indices;
    
    uint64_t vl = svcntb();
    
    if (rank == 0) {
        printf("============================================================\n");
#ifdef USE_MPI
        printf("SVE Gather/Scatter Bandwidth Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("SVE Gather/Scatter Bandwidth Benchmark\n");
#endif
        printf("============================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("Buffer Size: %lu MB per array\n", buffer_size / (1024 * 1024));
        printf("Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("Index Pool Size: %lu elements\n", index_pool_size);
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Registered Tests: %d\n\n", test_count);
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
    
    srand(42);
    uint64_t max_element_idx_64 = buffer_size / sizeof(int64_t) - 1;
    uint64_t max_idx = (max_element_idx_64 < INT32_MAX) ? max_element_idx_64 : INT32_MAX;
    
    uint64_t min_idx = max_idx, max_found = 0;
    uint64_t coverage_buckets = (max_idx / 64) + 2;
    uint64_t *coverage = (uint64_t *)calloc(coverage_buckets, sizeof(uint64_t));
    
    const char *mode_names[] = {"Random", "Uniform", "Hotspot", "RandomUniqueSorted"};
    
    if (index_mode == 0) {
        for (uint64_t i = 0; i < index_pool_size; i++) {
            uint64_t idx = ((uint64_t)rand() << 32 | rand()) % (max_idx + 1);
            gather_indices[i] = (int32_t)idx;
            update_index_stats(idx, &min_idx, &max_found, coverage);
        }
    } else if (index_mode == 1) {
        uint64_t stride = (max_idx + 1) / index_pool_size;
        if (stride == 0) stride = 1;
        uint64_t remainder = (max_idx + 1) - stride * index_pool_size;
        for (uint64_t i = 0; i < index_pool_size; i++) {
            uint64_t base = i * stride + (i < remainder ? i : remainder);
            uint64_t idx = base + (((uint64_t)rand() << 32 | rand()) % stride);
            if (idx > max_idx) idx = max_idx;
            gather_indices[i] = (int32_t)idx;
            update_index_stats(idx, &min_idx, &max_found, coverage);
        }
    } else if (index_mode == 2) {
        uint64_t hotspot_size = (max_idx + 1) / 10;
        uint64_t hotspot_start = (uint64_t)rand() % (max_idx + 1 - hotspot_size);
        for (uint64_t i = 0; i < index_pool_size; i++) {
            uint64_t idx = (rand() % 100 < 80) ? 
                hotspot_start + ((uint64_t)rand() % hotspot_size) :
                ((uint64_t)rand() << 32 | rand()) % (max_idx + 1);
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
            gather_indices[i] = (int32_t)unique_indices[i];
        }
        free(unique_indices);
    }
    
    uint64_t covered = 0;
    for (uint64_t i = 0; i < coverage_buckets - 1; i++) {
        covered += __builtin_popcountll(coverage[i]);
    }
    free(coverage);
    
    if (rank == 0) {
        printf("Index Mode: %s\n", mode_names[index_mode]);
        printf("Max Index: %lu (buffer elements: %lu)\n", max_idx, max_element_idx_64);
        printf("Generated Range: [%lu, %lu]\n", min_idx, max_found);
        printf("Unique Indices: %lu / %lu (%.2f%%)\n", covered, index_pool_size, 
               (double)covered / index_pool_size * 100.0);
        printf("Coverage: %.4f%% of buffer\n\n", 
               (double)covered / (max_idx + 1) * 100.0);
    }
    
    if (rank == 0) {
#ifdef USE_MPI
        printf("%-22s %15s %10s %10s %10s %10s\n", 
               "Test", "Category", "GB/s", "Time(ms)", "Data(MB)", "Total(GB/s)");
#else
        printf("%-22s %15s %10s %10s %10s\n", 
               "Test", "Category", "GB/s", "Time(ms)", "Data(MB)");
#endif
        printf("============================================================\n");
    }
    
    for (int i = 0; i < test_count; i++) {
        if (!run_all && !should_run_test(i, num_specs, specs)) continue;
        
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
        
        int verify_result = 0;
        if (test->func == sve_gather_ld1w_ld1w) {
            verify_result = verify_gather(a, b, 0);
        } else if (test->func == sve_gather_ld1sw_ld1d) {
            verify_result = verify_gather(a, c, 1);
        } else if (test->func == sve_scatter_st1w) {
            verify_result = verify_scatter(a, b, 0);
        } else if (test->func == sve_scatter_st1d) {
            verify_result = verify_scatter(a, b, 1);
        } else if (test->func == sve_gather_scatter_w) {
            verify_result = verify_gather_scatter(a, b, 0);
        } else if (test->func == sve_gather_scatter_d) {
            verify_result = verify_gather_scatter(a, b, 1);
        }
        
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        
        if (rank == 0 || print_all_ranks) {
#ifdef USE_MPI
            if (print_all_ranks) {
                printf("[Rank %d] %-22s %15s %10.2f %10.3f %10.0f",
                       rank, test->name, test->category, bandwidth, time_sec * 1000,
                       (double)bytes_per_iter / (1024 * 1024));
            } else {
                printf("%-22s %15s %10.2f %10.3f %10.0f %10.2f",
                       test->name, test->category, bandwidth, time_sec * 1000,
                       (double)bytes_per_iter / (1024 * 1024), total_bw);
            }
#else
            printf("%-22s %15s %10.2f %10.3f %10.0f",
                   test->name, test->category, bandwidth, time_sec * 1000,
                   (double)bytes_per_iter / (1024 * 1024));
#endif
            if (verify_result > 0) {
                printf("  VERIFY_FAIL(%d)", verify_result);
            }
            printf("\n");
        }
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    if (rank == 0) {
        printf("============================================================\n");
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