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

#define BUFFER_SIZE (128 * 1024 * 1024)
#define WARMUP_ITER 5
#define TEST_ITER 10
#define INDEX_POOL_SIZE (1024 * 1024)

static int32_t *gather_indices = NULL;

typedef struct {
    const char *name;
    const char *category;
    void (*func)(void *a, void *b, void *c, uint64_t size, double scalar);
    uint64_t bytes_per_iter;
    int requires_c;
    int requires_scalar;
} test_item_t;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

#pragma region NEON_LOAD_TESTS

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

#pragma endregion

#pragma region NEON_STORE_TESTS

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

#pragma endregion

#pragma region NEON_COPY_TESTS

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

#pragma endregion

#pragma region SVE_LD1B_TESTS

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

#pragma endregion

#pragma region SVE_ST1B_TESTS

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

#pragma endregion

#pragma region SVE_LD1B_ST1B_TESTS

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

#pragma endregion

#pragma region SVE_LD1W_TESTS

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

#pragma endregion

#pragma region SVE_ST1W_TESTS

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

#pragma endregion

#pragma region SVE_LD1W_ST1W_TESTS

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

#pragma endregion

#pragma region SVE_LD1D_TESTS

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

#pragma endregion

#pragma region SVE_ST1D_TESTS

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

#pragma endregion

#pragma region SVE_LD1D_ST1D_TESTS

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

#pragma endregion

#pragma region STREAM_BENCHMARK_TESTS

static void stream_copy(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *dst = (double *)a;
    double *src = (double *)b;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 4;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "ld1d z0.d, p0/z, [%[s], #0, MUL VL]\n"
            "ld1d z1.d, p0/z, [%[s], #1, MUL VL]\n"
            "ld1d z2.d, p0/z, [%[s], #2, MUL VL]\n"
            "ld1d z3.d, p0/z, [%[s], #3, MUL VL]\n"
            "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
            "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
            "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
            "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "memory"
        );
    }
}

static void stream_scale(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *dst = (double *)a;
    double *src = (double *)b;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 4;
    uint64_t chunks = size / chunk_size;
    svfloat64_t scale_vec = svdup_f64(scalar);
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "ld1d z0.d, p0/z, [%[s], #0, MUL VL]\n"
            "ld1d z1.d, p0/z, [%[s], #1, MUL VL]\n"
            "ld1d z2.d, p0/z, [%[s], #2, MUL VL]\n"
            "ld1d z3.d, p0/z, [%[s], #3, MUL VL]\n"
            "fmul z0.d, p0/m, z0.d, %[k].d\n"
            "fmul z1.d, p0/m, z1.d, %[k].d\n"
            "fmul z2.d, p0/m, z2.d, %[k].d\n"
            "fmul z3.d, p0/m, z3.d, %[k].d\n"
            "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
            "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
            "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
            "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size), [k] "w" (scale_vec)
            : "p0", "z0", "z1", "z2", "z3", "memory"
        );
    }
}

static void stream_add(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *dst = (double *)a;
    double *src1 = (double *)b;
    double *src2 = (double *)c;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 4;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "ld1d z0.d, p0/z, [%[s1], #0, MUL VL]\n"
            "ld1d z1.d, p0/z, [%[s1], #1, MUL VL]\n"
            "ld1d z2.d, p0/z, [%[s1], #2, MUL VL]\n"
            "ld1d z3.d, p0/z, [%[s1], #3, MUL VL]\n"
            "ld1d z4.d, p0/z, [%[s2], #0, MUL VL]\n"
            "ld1d z5.d, p0/z, [%[s2], #1, MUL VL]\n"
            "ld1d z6.d, p0/z, [%[s2], #2, MUL VL]\n"
            "ld1d z7.d, p0/z, [%[s2], #3, MUL VL]\n"
            "fadd z0.d, p0/m, z0.d, z4.d\n"
            "fadd z1.d, p0/m, z1.d, z5.d\n"
            "fadd z2.d, p0/m, z2.d, z6.d\n"
            "fadd z3.d, p0/m, z3.d, z7.d\n"
            "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
            "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
            "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
            "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
            "add %[s1], %[s1], %[inc]\n"
            "add %[s2], %[s2], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s1] "+r" (src1), [s2] "+r" (src2), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

static void stream_triad(void *a, void *b, void *c, uint64_t size, double scalar) {
    double *dst = (double *)a;
    double *src1 = (double *)b;
    double *src2 = (double *)c;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 4;
    uint64_t chunks = size / chunk_size;
    svfloat64_t scale_vec = svdup_f64(scalar);
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.d\n"
            "ld1d z0.d, p0/z, [%[s1], #0, MUL VL]\n"
            "ld1d z1.d, p0/z, [%[s1], #1, MUL VL]\n"
            "ld1d z2.d, p0/z, [%[s1], #2, MUL VL]\n"
            "ld1d z3.d, p0/z, [%[s1], #3, MUL VL]\n"
            "ld1d z4.d, p0/z, [%[s2], #0, MUL VL]\n"
            "ld1d z5.d, p0/z, [%[s2], #1, MUL VL]\n"
            "ld1d z6.d, p0/z, [%[s2], #2, MUL VL]\n"
            "ld1d z7.d, p0/z, [%[s2], #3, MUL VL]\n"
            "fmul z0.d, p0/m, z0.d, %[k].d\n"
            "fmul z1.d, p0/m, z1.d, %[k].d\n"
            "fmul z2.d, p0/m, z2.d, %[k].d\n"
            "fmul z3.d, p0/m, z3.d, %[k].d\n"
            "fadd z0.d, p0/m, z0.d, z4.d\n"
            "fadd z1.d, p0/m, z1.d, z5.d\n"
            "fadd z2.d, p0/m, z2.d, z6.d\n"
            "fadd z3.d, p0/m, z3.d, z7.d\n"
            "st1d z0.d, p0, [%[d], #0, MUL VL]\n"
            "st1d z1.d, p0, [%[d], #1, MUL VL]\n"
            "st1d z2.d, p0, [%[d], #2, MUL VL]\n"
            "st1d z3.d, p0, [%[d], #3, MUL VL]\n"
            "add %[s1], %[s1], %[inc]\n"
            "add %[s2], %[s2], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s1] "+r" (src1), [s2] "+r" (src2), [d] "+r" (dst)
            : [inc] "r" (chunk_size), [k] "w" (scale_vec)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

#pragma endregion

#pragma region SVE_GATHER_TESTS

static void sve_gather_ld1w_ld1w(void *a, void *b, void *c, uint64_t size, double scalar) {
    float *src = (float *)b;
    float *dst = (float *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl = svcntb() / sizeof(int32_t);
    uint64_t chunk_bytes = vl * 8 * sizeof(int32_t);
    uint64_t iterations = size / chunk_bytes;
    uint64_t idx_pool_iters = INDEX_POOL_SIZE / (vl * 8);
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (i % idx_pool_iters == 0) idx_base = gather_indices;
        
        __asm__ volatile (
            "ptrue p0.s\n"
            "ld1w z8.s, p0/z, [%[idx], #0, MUL VL]\n"
            "ld1w z9.s, p0/z, [%[idx], #1, MUL VL]\n"
            "ld1w z10.s, p0/z, [%[idx], #2, MUL VL]\n"
            "ld1w z11.s, p0/z, [%[idx], #3, MUL VL]\n"
            "ld1w z0.s, p0/z, [%[s], z8.s, sxtw 2]\n"
            "ld1w z1.s, p0/z, [%[s], z9.s, sxtw 2]\n"
            "ld1w z2.s, p0/z, [%[s], z10.s, sxtw 2]\n"
            "ld1w z3.s, p0/z, [%[s], z11.s, sxtw 2]\n"
            "ld1w z12.s, p0/z, [%[idx], #4, MUL VL]\n"
            "ld1w z13.s, p0/z, [%[idx], #5, MUL VL]\n"
            "ld1w z14.s, p0/z, [%[idx], #6, MUL VL]\n"
            "ld1w z15.s, p0/z, [%[idx], #7, MUL VL]\n"
            "ld1w z4.s, p0/z, [%[s], z12.s, sxtw 2]\n"
            "ld1w z5.s, p0/z, [%[s], z13.s, sxtw 2]\n"
            "ld1w z6.s, p0/z, [%[s], z14.s, sxtw 2]\n"
            "ld1w z7.s, p0/z, [%[s], z15.s, sxtw 2]\n"
            "st1w z0.s, p0, [%[d], #0, MUL VL]\n"
            "st1w z1.s, p0, [%[d], #1, MUL VL]\n"
            "st1w z2.s, p0, [%[d], #2, MUL VL]\n"
            "st1w z3.s, p0, [%[d], #3, MUL VL]\n"
            "st1w z4.s, p0, [%[d], #4, MUL VL]\n"
            "st1w z5.s, p0, [%[d], #5, MUL VL]\n"
            "st1w z6.s, p0, [%[d], #6, MUL VL]\n"
            "st1w z7.s, p0, [%[d], #7, MUL VL]\n"
            "add %[idx], %[idx], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [idx] "+r" (idx_base), [d] "+r" (dst)
            : [s] "r" (src), [inc] "r" (chunk_bytes)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "memory"
        );
    }
}

static void sve_gather_ld1sw_ld1d(void *a, void *b, void *c, uint64_t size, double scalar) {
    int32_t *src_sw = (int32_t *)b;
    double *src_d = (double *)c;
    double *dst = (double *)a;
    int32_t *idx_base = gather_indices;
    uint64_t vl_bytes = svcntb();
    uint64_t vl_d = vl_bytes / sizeof(int64_t);
    uint64_t vl_s = vl_bytes / sizeof(int32_t);
    uint64_t chunk_bytes = vl_d * 4 * sizeof(double);
    uint64_t iterations = size / chunk_bytes;
    uint64_t idx_pool_iters = INDEX_POOL_SIZE / (vl_s * 4);
    
    for (uint64_t i = 0; i < iterations; i++) {
        if (i % idx_pool_iters == 0) idx_base = gather_indices;
        
        __asm__ volatile (
            "ptrue p0.s\n"
            "ld1w z0.s, p0/z, [%[idx], #0, MUL VL]\n"
            "ld1w z1.s, p0/z, [%[idx], #1, MUL VL]\n"
            "ld1w z2.s, p0/z, [%[idx], #2, MUL VL]\n"
            "ld1w z3.s, p0/z, [%[idx], #3, MUL VL]\n"
            "ptrue p1.d\n"
            "sunpklo z4.d, z0.s\n"
            "sunpklo z5.d, z1.s\n"
            "sunpklo z6.d, z2.s\n"
            "sunpklo z7.d, z3.s\n"
            "ld1sw z8.d, p1/z, [%[sw], z4.d, lsl 2]\n"
            "ld1sw z9.d, p1/z, [%[sw], z5.d, lsl 2]\n"
            "ld1sw z10.d, p1/z, [%[sw], z6.d, lsl 2]\n"
            "ld1sw z11.d, p1/z, [%[sw], z7.d, lsl 2]\n"
            "ld1d z12.d, p1/z, [%[sd], z4.d, lsl 3]\n"
            "ld1d z13.d, p1/z, [%[sd], z5.d, lsl 3]\n"
            "ld1d z14.d, p1/z, [%[sd], z6.d, lsl 3]\n"
            "ld1d z15.d, p1/z, [%[sd], z7.d, lsl 3]\n"
            "st1d z12.d, p1, [%[d], #0, MUL VL]\n"
            "st1d z13.d, p1, [%[d], #1, MUL VL]\n"
            "st1d z14.d, p1, [%[d], #2, MUL VL]\n"
            "st1d z15.d, p1, [%[d], #3, MUL VL]\n"
            "add %[idx], %[idx], %[inci]\n"
            "add %[d], %[d], %[incd]\n"
            : [idx] "+r" (idx_base), [d] "+r" (dst)
            : [sw] "r" (src_sw), [sd] "r" (src_d), 
              [inci] "r" (vl_s * 4 * sizeof(int32_t)),
              [incd] "r" (vl_d * 4 * sizeof(double))
            : "p0", "p1",
              "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "memory"
        );
    }
}

#pragma endregion

#pragma GCC pop_options

#pragma region TEST_REGISTRY

static test_item_t test_registry[] = {
    {"NEON LDP (Read)",      "Load",    neon_ldp_read,        BUFFER_SIZE,      0, 0},
    {"NEON STP (Write)",     "Store",   neon_stp_write,       BUFFER_SIZE,      0, 0},
    {"NEON LDP+STP (Copy)",  "Copy",    neon_ldp_stp_copy,    BUFFER_SIZE * 2,  0, 0},
    
    {"SVE LD1B (Read)",      "Load",    sve_ld1b_read,        BUFFER_SIZE,      0, 0},
    {"SVE ST1B (Write)",     "Store",   sve_st1b_write,       BUFFER_SIZE,      0, 0},
    {"SVE LD1B+ST1B (Copy)", "Copy",    sve_ld1b_st1b_copy,   BUFFER_SIZE * 2,  0, 0},
    
    {"SVE LD1W (Read)",      "Load",    sve_ld1w_read,        BUFFER_SIZE,      0, 0},
    {"SVE ST1W (Write)",     "Store",   sve_st1w_write,       BUFFER_SIZE,      0, 0},
    {"SVE LD1W+ST1W (Copy)", "Copy",    sve_ld1w_st1w_copy,   BUFFER_SIZE * 2,  0, 0},
    
    {"SVE LD1D (Read)",      "Load",    sve_ld1d_read,        BUFFER_SIZE,      0, 0},
    {"SVE ST1D (Write)",     "Store",   sve_st1d_write,       BUFFER_SIZE,      0, 0},
    {"SVE LD1D+ST1D (Copy)", "Copy",    sve_ld1d_st1d_copy,   BUFFER_SIZE * 2,  0, 0},
    
    {"SVE Gather LD1W",      "Gather",  sve_gather_ld1w_ld1w, BUFFER_SIZE * 2,  0, 0},
    {"SVE Gather LD1SW+LD1D","Gather",  sve_gather_ld1sw_ld1d,BUFFER_SIZE * 3,  1, 0},
    
    {"STREAM Copy",          "STREAM",  stream_copy,          BUFFER_SIZE * 2,  0, 0},
    {"STREAM Scale",         "STREAM",  stream_scale,         BUFFER_SIZE * 2,  0, 1},
    {"STREAM Add",           "STREAM",  stream_add,           BUFFER_SIZE * 3,  1, 0},
    {"STREAM Triad",         "STREAM",  stream_triad,         BUFFER_SIZE * 3,  1, 1},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

#pragma endregion

static int verify_gather_ld1w_ld1w(void *a, void *b, void *c, int rank) {
    float *src = (float *)b;
    float *dst = (float *)a;
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl = svcntb() / sizeof(int32_t);
    uint64_t total_elements = BUFFER_SIZE / sizeof(float);
    
    for (uint64_t i = 0; i < total_elements && errors < 5; i++) {
        uint64_t iter = i / (vl * 8);
        uint64_t pos_in_iter = i % (vl * 8);
        uint64_t idx_pos = (iter % (INDEX_POOL_SIZE / (vl * 8))) * (vl * 8) + pos_in_iter;
        
        if (idx_pos >= INDEX_POOL_SIZE) continue;
        
        int32_t src_elem_idx = indices[idx_pos];
        float expected = src[src_elem_idx];
        float actual = dst[i];
        
        if (expected != actual) {
            if (errors == 0) {
                fprintf(stderr, "[Rank %d] LD1W Gather verify FAILED:\n", rank);
            }
            fprintf(stderr, "  dst[%lu]: expected %.1f (src[%d]), got %.1f\n",
                    i, expected, src_elem_idx, actual);
            errors++;
        }
    }
    return errors;
}

static int verify_gather_ld1sw_ld1d(void *a, void *b, void *c, int rank) {
    double *src_d = (double *)c;
    double *dst = (double *)a;
    int32_t *indices = gather_indices;
    int errors = 0;
    uint64_t vl_d = svcntb() / sizeof(int64_t);
    uint64_t vl_s = svcntb() / sizeof(int32_t);
    uint64_t total_elements = BUFFER_SIZE / sizeof(double);
    
    for (uint64_t i = 0; i < total_elements && errors < 5; i++) {
        uint64_t iter = i / (vl_d * 4);
        uint64_t pos_in_iter = i % (vl_d * 4);
        uint64_t idx_pos = (iter % (INDEX_POOL_SIZE / (vl_s * 4))) * (vl_s * 4) + pos_in_iter;
        
        if (idx_pos >= INDEX_POOL_SIZE) continue;
        
        int32_t src_elem_idx = indices[idx_pos];
        double expected = src_d[src_elem_idx];
        double actual = dst[i];
        
        if (expected != actual) {
            if (errors == 0) {
                fprintf(stderr, "[Rank %d] LD1SW+LD1D Gather verify FAILED:\n", rank);
            }
            fprintf(stderr, "  dst[%lu]: expected %.1f (src_d[%d]), got %.1f\n",
                    i, expected, src_elem_idx, actual);
            errors++;
        }
    }
    return errors;
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
    
    for (int i = 0; i < WARMUP_ITER; i++) {
        test->func(a, b, c, BUFFER_SIZE, scalar);
    }
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < TEST_ITER; i++) {
        test->func(a, b, c, BUFFER_SIZE, scalar);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return time_sec / TEST_ITER;
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
    
    uint64_t vl = svcntb();
    
    if (rank == 0) {
        printf("============================================================\n");
#ifdef USE_MPI
        printf("SVE Bandwidth Benchmark (MPI Parallel - %d processes)\n", nprocs);
#else
        printf("SVE Bandwidth Benchmark (Single Process)\n");
#endif
        printf("============================================================\n");
        printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
        printf("Buffer Size: %d MB per array\n", BUFFER_SIZE / (1024 * 1024));
        printf("Warmup Iterations: %d\n", WARMUP_ITER);
        printf("Test Iterations: %d\n", TEST_ITER);
        printf("Registered Tests: %d\n\n", test_count);
    }
    
    void *a = NULL, *b = NULL, *c = NULL;
    
    if (posix_memalign(&a, 64, BUFFER_SIZE) != 0 ||
        posix_memalign(&b, 64, BUFFER_SIZE) != 0 ||
        posix_memalign(&c, 64, BUFFER_SIZE) != 0 ||
        posix_memalign((void**)&gather_indices, 64, INDEX_POOL_SIZE * sizeof(int32_t)) != 0) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate aligned memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate aligned memory\n");
#endif
        return 1;
    }
    
    memset(a, 0x55, BUFFER_SIZE);
    memset(b, 0xAA, BUFFER_SIZE);
    memset(c, 0x33, BUFFER_SIZE);
    
    srand(42 + rank);
    uint64_t max_element_idx_64 = BUFFER_SIZE / sizeof(int64_t) - 1;
    uint64_t stride = max_element_idx_64 / INDEX_POOL_SIZE;
    if (stride < 1) stride = 1;
    for (uint64_t i = 0; i < INDEX_POOL_SIZE; i++) {
        gather_indices[i] = (int32_t)((i * stride) + (rand() % stride));
    }
    
    double *da = (double *)a;
    double *db = (double *)b;
    double *dc = (double *)c;
    for (uint64_t i = 0; i < BUFFER_SIZE / sizeof(double); i++) {
        da[i] = 1.0;
        db[i] = 2.0;
        dc[i] = 3.0;
    }
    
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
        test_item_t *test = &test_registry[i];
        
#ifdef USE_MPI
        double time_sec = run_test(test, a, b, c, MPI_COMM_WORLD);
#else
        double time_sec = run_test(test, a, b, c);
#endif
        double local_bw = get_bandwidth(test->bytes_per_iter, time_sec);
        
#ifdef USE_MPI
        double total_bw = 0.0;
        MPI_Reduce(&local_bw, &total_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
        
        int verify_result = 0;
        if (test->func == sve_gather_ld1w_ld1w) {
            verify_result = verify_gather_ld1w_ld1w(a, b, c, rank);
        } else if (test->func == sve_gather_ld1sw_ld1d) {
            verify_result = verify_gather_ld1sw_ld1d(a, b, c, rank);
        }
        
        if (rank == 0) {
#ifdef USE_MPI
            printf("%-22s %10s %10.2f %10.3f %10.0f %10.2f",
                   test->name, test->category, local_bw, time_sec * 1000,
                   (double)test->bytes_per_iter / (1024 * 1024), total_bw);
#else
            printf("%-22s %10s %10.2f %10.3f %10.0f",
                   test->name, test->category, local_bw, time_sec * 1000,
                   (double)test->bytes_per_iter / (1024 * 1024));
#endif
            if (verify_result > 0) {
                printf("  VERIFY_FAIL(%d)", verify_result);
            }
            printf("\n");
        }
    }
    
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