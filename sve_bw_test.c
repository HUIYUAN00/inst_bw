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

typedef struct {
    const char *name;
    const char *category;
    void (*func)(void *a, void *b, void *c, uint64_t size, double scalar);
    uint64_t bytes_per_iter;
} test_item_t;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

//=== NEON_TESTS

static void neon_ldp_read(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *src = (float *)a;
    uint64_t chunks = size / 256;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ldp q0, q1, [%[s], #0]\n"
            "ldp q2, q3, [%[s], #32]\n"
            "ldp q4, q5, [%[s], #64]\n"
            "ldp q6, q7, [%[s], #96]\n"
            "ldp q8, q9, [%[s], #128]\n"
            "ldp q10, q11, [%[s], #160]\n"
            "ldp q12, q13, [%[s], #192]\n"
            "ldp q14, q15, [%[s], #224]\n"
            "add %[s], %[s], #256\n"
            : [s] "+r" (src)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "memory"
        );
    }
}

static void neon_stp_write(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *dst = (float *)a;
    uint64_t chunks = size / 256;
    __asm__ volatile (
        "movi v0.4s, #1\n"
        "movi v1.4s, #2\n"
        "movi v2.4s, #3\n"
        "movi v3.4s, #4\n"
        "movi v4.4s, #5\n"
        "movi v5.4s, #6\n"
        "movi v6.4s, #7\n"
        "movi v7.4s, #8\n"
        "movi v8.4s, #9\n"
        "movi v9.4s, #10\n"
        "movi v10.4s, #11\n"
        "movi v11.4s, #12\n"
        "movi v12.4s, #13\n"
        "movi v13.4s, #14\n"
        "movi v14.4s, #15\n"
        "movi v15.4s, #16\n"
        :
        :
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
          "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
    );
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "stp q0, q1, [%[d], #0]\n"
            "stp q2, q3, [%[d], #32]\n"
            "stp q4, q5, [%[d], #64]\n"
            "stp q6, q7, [%[d], #96]\n"
            "stp q8, q9, [%[d], #128]\n"
            "stp q10, q11, [%[d], #160]\n"
            "stp q12, q13, [%[d], #192]\n"
            "stp q14, q15, [%[d], #224]\n"
            "add %[d], %[d], #256\n"
            : [d] "+r" (dst)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "memory"
        );
    }
}

static void neon_ldp_stp_copy(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *src = (float *)b;
    float *dst = (float *)a;
    uint64_t chunks = size / 128;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ldp q0, q1, [%[s], #0]\n"
            "ldp q2, q3, [%[s], #32]\n"
            "ldp q4, q5, [%[s], #64]\n"
            "ldp q6, q7, [%[s], #96]\n"
            "stp q0, q1, [%[d], #0]\n"
            "stp q2, q3, [%[d], #32]\n"
            "stp q4, q5, [%[d], #64]\n"
            "stp q6, q7, [%[d], #96]\n"
            "add %[s], %[s], #128\n"
            "add %[d], %[d], #128\n"
            : [s] "+r" (src), [d] "+r" (dst)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "memory"
        );
    }
}

//=== End

//=== SVE_LD1B_TESTS

static void sve_ld1b_read(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *src = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.b\n"
            "ld1b z0.b, p0/z, [%[s], #0, MUL VL]\n"
            "ld1b z1.b, p0/z, [%[s], #1, MUL VL]\n"
            "ld1b z2.b, p0/z, [%[s], #2, MUL VL]\n"
            "ld1b z3.b, p0/z, [%[s], #3, MUL VL]\n"
            "ld1b z4.b, p0/z, [%[s], #4, MUL VL]\n"
            "ld1b z5.b, p0/z, [%[s], #5, MUL VL]\n"
            "ld1b z6.b, p0/z, [%[s], #6, MUL VL]\n"
            "ld1b z7.b, p0/z, [%[s], #7, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            : [s] "+r" (src)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void sve_st1b_write(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *dst = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    __asm__ volatile (
        "ptrue p0.b\n"
        "mov z0.b, #1\n"
        "mov z1.b, #2\n"
        "mov z2.b, #3\n"
        "mov z3.b, #4\n"
        "mov z4.b, #5\n"
        "mov z5.b, #6\n"
        "mov z6.b, #7\n"
        "mov z7.b, #8\n"
        :
        :
        : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"
    );
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.b\n"
            "st1b z0.b, p0, [%[d], #0, MUL VL]\n"
            "st1b z1.b, p0, [%[d], #1, MUL VL]\n"
            "st1b z2.b, p0, [%[d], #2, MUL VL]\n"
            "st1b z3.b, p0, [%[d], #3, MUL VL]\n"
            "st1b z4.b, p0, [%[d], #4, MUL VL]\n"
            "st1b z5.b, p0, [%[d], #5, MUL VL]\n"
            "st1b z6.b, p0, [%[d], #6, MUL VL]\n"
            "st1b z7.b, p0, [%[d], #7, MUL VL]\n"
            "add %[d], %[d], %[inc]\n"
            : [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void sve_ld1b_st1b_copy(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *src = (uint8_t *)b;
    uint8_t *dst = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.b\n"
            "ld1b z0.b, p0/z, [%[s], #0, MUL VL]\n"
            "ld1b z1.b, p0/z, [%[s], #1, MUL VL]\n"
            "ld1b z2.b, p0/z, [%[s], #2, MUL VL]\n"
            "ld1b z3.b, p0/z, [%[s], #3, MUL VL]\n"
            "ld1b z4.b, p0/z, [%[s], #4, MUL VL]\n"
            "ld1b z5.b, p0/z, [%[s], #5, MUL VL]\n"
            "ld1b z6.b, p0/z, [%[s], #6, MUL VL]\n"
            "ld1b z7.b, p0/z, [%[s], #7, MUL VL]\n"
            "st1b z0.b, p0, [%[d], #0, MUL VL]\n"
            "st1b z1.b, p0, [%[d], #1, MUL VL]\n"
            "st1b z2.b, p0, [%[d], #2, MUL VL]\n"
            "st1b z3.b, p0, [%[d], #3, MUL VL]\n"
            "st1b z4.b, p0, [%[d], #4, MUL VL]\n"
            "st1b z5.b, p0, [%[d], #5, MUL VL]\n"
            "st1b z6.b, p0, [%[d], #6, MUL VL]\n"
            "st1b z7.b, p0, [%[d], #7, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

//=== End

//=== SVE_LD1W_TESTS

static void sve_ld1w_read(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *src = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.s\n"
            "ld1w z0.s, p0/z, [%[s], #0, MUL VL]\n"
            "ld1w z1.s, p0/z, [%[s], #1, MUL VL]\n"
            "ld1w z2.s, p0/z, [%[s], #2, MUL VL]\n"
            "ld1w z3.s, p0/z, [%[s], #3, MUL VL]\n"
            "ld1w z4.s, p0/z, [%[s], #4, MUL VL]\n"
            "ld1w z5.s, p0/z, [%[s], #5, MUL VL]\n"
            "ld1w z6.s, p0/z, [%[s], #6, MUL VL]\n"
            "ld1w z7.s, p0/z, [%[s], #7, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            : [s] "+r" (src)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void sve_st1w_write(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *dst = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    __asm__ volatile (
        "ptrue p0.s\n"
        "mov z0.s, #1\n"
        "mov z1.s, #2\n"
        "mov z2.s, #3\n"
        "mov z3.s, #4\n"
        "mov z4.s, #5\n"
        "mov z5.s, #6\n"
        "mov z6.s, #7\n"
        "mov z7.s, #8\n"
        :
        :
        : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"
    );
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.s\n"
            "st1w z0.s, p0, [%[d], #0, MUL VL]\n"
            "st1w z1.s, p0, [%[d], #1, MUL VL]\n"
            "st1w z2.s, p0, [%[d], #2, MUL VL]\n"
            "st1w z3.s, p0, [%[d], #3, MUL VL]\n"
            "st1w z4.s, p0, [%[d], #4, MUL VL]\n"
            "st1w z5.s, p0, [%[d], #5, MUL VL]\n"
            "st1w z6.s, p0, [%[d], #6, MUL VL]\n"
            "st1w z7.s, p0, [%[d], #7, MUL VL]\n"
            "add %[d], %[d], %[inc]\n"
            : [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void sve_ld1w_st1w_copy(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *src = (uint8_t *)b;
    uint8_t *dst = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.s\n"
            "ld1w z0.s, p0/z, [%[s], #0, MUL VL]\n"
            "ld1w z1.s, p0/z, [%[s], #1, MUL VL]\n"
            "ld1w z2.s, p0/z, [%[s], #2, MUL VL]\n"
            "ld1w z3.s, p0/z, [%[s], #3, MUL VL]\n"
            "ld1w z4.s, p0/z, [%[s], #4, MUL VL]\n"
            "ld1w z5.s, p0/z, [%[s], #5, MUL VL]\n"
            "ld1w z6.s, p0/z, [%[s], #6, MUL VL]\n"
            "ld1w z7.s, p0/z, [%[s], #7, MUL VL]\n"
            "st1w z0.s, p0, [%[d], #0, MUL VL]\n"
            "st1w z1.s, p0, [%[d], #1, MUL VL]\n"
            "st1w z2.s, p0, [%[d], #2, MUL VL]\n"
            "st1w z3.s, p0, [%[d], #3, MUL VL]\n"
            "st1w z4.s, p0, [%[d], #4, MUL VL]\n"
            "st1w z5.s, p0, [%[d], #5, MUL VL]\n"
            "st1w z6.s, p0, [%[d], #6, MUL VL]\n"
            "st1w z7.s, p0, [%[d], #7, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

//=== End

//=== SVE_LD1D_TESTS

static void sve_ld1d_read(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *src = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "ld1d z0.d, p0/z, [%[s], #0, MUL VL]\n"
            "ld1d z1.d, p0/z, [%[s], #1, MUL VL]\n"
            "ld1d z2.d, p0/z, [%[s], #2, MUL VL]\n"
            "ld1d z3.d, p0/z, [%[s], #3, MUL VL]\n"
            "ld1d z4.d, p0/z, [%[s], #4, MUL VL]\n"
            "ld1d z5.d, p0/z, [%[s], #5, MUL VL]\n"
            "ld1d z6.d, p0/z, [%[s], #6, MUL VL]\n"
            "ld1d z7.d, p0/z, [%[s], #7, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            : [s] "+r" (src)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void sve_st1d_write(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *dst = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    __asm__ volatile (
        "ptrue p0.d\n"
        "mov z0.d, #1\n"
        "mov z1.d, #2\n"
        "mov z2.d, #3\n"
        "mov z3.d, #4\n"
        "mov z4.d, #5\n"
        "mov z5.d, #6\n"
        "mov z6.d, #7\n"
        "mov z7.d, #8\n"
        :
        :
        : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"
    );
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
            "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
            "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
            "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
            "st1d z4.d, p0, [%[d], #4, MUL VL]\n"
            "st1d z5.d, p0, [%[d], #5, MUL VL]\n"
            "st1d z6.d, p0, [%[d], #6, MUL VL]\n"
            "st1d z7.d, p0, [%[d], #7, MUL VL]\n"
            "add %[d], %[d], %[inc]\n"
            : [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void sve_ld1d_st1d_copy(void *a, void *b, void *c, uint64_t size, double scalar) {
    uint8_t *src = (uint8_t *)b;
    uint8_t *dst = (uint8_t *)a;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "ld1d z0.d, p0/z, [%[s], #0, MUL VL]\n"
            "ld1d z1.d, p0/z, [%[s], #1, MUL VL]\n"
            "ld1d z2.d, p0/z, [%[s], #2, MUL VL]\n"
            "ld1d z3.d, p0/z, [%[s], #3, MUL VL]\n"
            "ld1d z4.d, p0/z, [%[s], #4, MUL VL]\n"
            "ld1d z5.d, p0/z, [%[s], #5, MUL VL]\n"
            "ld1d z6.d, p0/z, [%[s], #6, MUL VL]\n"
            "ld1d z7.d, p0/z, [%[s], #7, MUL VL]\n"
            "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
            "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
            "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
            "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
            "st1d z4.d, p0, [%[d], #4, MUL VL]\n"
            "st1d z5.d, p0, [%[d], #5, MUL VL]\n"
            "st1d z6.d, p0, [%[d], #6, MUL VL]\n"
            "st1d z7.d, p0, [%[d], #7, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

//=== End

#pragma GCC pop_options

//=== TEST_REGISTRY

static test_item_t test_registry[] = {
    {"NEON LDP (Read)",      "Load",    neon_ldp_read,        0},
    {"NEON STP (Write)",     "Store",   neon_stp_write,       0},
    {"NEON LDP+STP (Copy)",  "Copy",    neon_ldp_stp_copy,    0},
    
    {"SVE LD1B (Read)",      "Load",    sve_ld1b_read,        0},
    {"SVE ST1B (Write)",     "Store",   sve_st1b_write,       0},
    {"SVE LD1B+ST1B (Copy)", "Copy",    sve_ld1b_st1b_copy,   0},
    
    {"SVE LD1W (Read)",      "Load",    sve_ld1w_read,        0},
    {"SVE ST1W (Write)",     "Store",   sve_st1w_write,       0},
    {"SVE LD1W+ST1W (Copy)", "Copy",    sve_ld1w_st1w_copy,   0},
    
    {"SVE LD1D (Read)",      "Load",    sve_ld1d_read,        0},
    {"SVE ST1D (Write)",     "Store",   sve_st1d_write,       0},
    {"SVE LD1D+ST1D (Copy)", "Copy",    sve_ld1d_st1d_copy,   0},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options] [test_spec...]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -l, --list              List all available tests\n");
    printf("  -b, --buffer-size <MB>  Buffer size in MB (default: 128)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("\nTest Specification:\n");
    printf("  <index>                 Run test by index (0-based)\n");
    printf("  <name>                  Run test by name (partial match)\n");
    printf("  <category>              Run all tests in a category\n");
    printf("\nCategories: Load, Store, Copy\n");
    printf("\nExamples:\n");
    printf("  %s                               Run all tests\n", prog_name);
    printf("  %s -b 64                         Use 64MB buffer\n", prog_name);
    printf("  %s Load                          Run all Load tests\n", prog_name);
    printf("  %s -b 256 Copy                   256MB buffer, run Copy tests\n", prog_name);
    printf("  %s 0 6 9                         Run tests 0, 6, and 9\n", prog_name);
}

static void print_tests(void) {
    printf("Available Tests:\n");
    printf("============================================================\n");
    printf("%-4s %-22s %10s\n", "Idx", "Test Name", "Category");
    printf("============================================================\n");
    for (int i = 0; i < test_count; i++) {
        printf("%-4d %-22s %10s\n", i, test_registry[i].name, test_registry[i].category);
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
        run_all = 0;
        num_specs++;
    }
    
    if (!run_all && num_specs > 0) {
        specs = &argv[argc - num_specs];
    }
    
#ifdef USE_MPI
    MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    
    uint64_t vl = svcntb();
    
    if (rank == 0) {
        printf("============================================================\n");
#ifdef USE_MPI
        printf("SVE Sequential Bandwidth Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("SVE Sequential Bandwidth Benchmark\n");
#endif
        printf("============================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("Buffer Size: %lu MB per array\n", buffer_size / (1024 * 1024));
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Registered Tests: %d\n\n", test_count);
    }
    
    void *a = NULL, *b = NULL, *c = NULL;
    
    if (posix_memalign(&a, 64, buffer_size) != 0 ||
        posix_memalign(&b, 64, buffer_size) != 0 ||
        posix_memalign(&c, 64, buffer_size) != 0) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate aligned memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate aligned memory\n");
#endif
        return 1;
    }
    
    memset(a, 0x55, buffer_size);
    memset(b, 0xAA, buffer_size);
    memset(c, 0x33, buffer_size);
    
    if (rank == 0) {
#ifdef USE_MPI
        printf("%-22s %10s %10s %10s %10s %10s\n", 
               "Test", "Category", "GB/s", "Time(ms)", "Data(MB)", "Total(GB/s)");
#else
        printf("%-22s %10s %10s %10s %10s\n", 
               "Test", "Category", "GB/s", "Time(ms)", "Data(MB)");
#endif
        printf("============================================================\n");
    }
    
    for (int i = 0; i < test_count; i++) {
        if (!run_all && !should_run_test(i, num_specs, specs)) continue;
        
        test_item_t *test = &test_registry[i];
        
        uint64_t bytes_per_iter = buffer_size;
        if (strcmp(test->category, "Copy") == 0) {
            bytes_per_iter = buffer_size * 2;
        }
        
#ifdef USE_MPI
        double time_sec = run_test(test, a, b, c, MPI_COMM_WORLD);
#else
        double time_sec = run_test(test, a, b, c);
#endif
        double local_bw = get_bandwidth(bytes_per_iter, time_sec);
        
#ifdef USE_MPI
        double total_bw = 0.0;
        MPI_Reduce(&local_bw, &total_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
        
        if (rank == 0) {
#ifdef USE_MPI
            printf("%-22s %10s %10.2f %10.3f %10.0f %10.2f\n",
                   test->name, test->category, local_bw, time_sec * 1000,
                   (double)bytes_per_iter / (1024 * 1024), total_bw);
#else
            printf("%-22s %10s %10.2f %10.3f %10.0f\n",
                   test->name, test->category, local_bw, time_sec * 1000,
                   (double)bytes_per_iter / (1024 * 1024));
#endif
        }
    }
    
    if (rank == 0) {
        printf("============================================================\n");
    }
    
    free(a);
    free(b);
    free(c);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}