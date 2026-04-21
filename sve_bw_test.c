#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_sve.h>
#include <sys/mman.h>

#define BUFFER_SIZE (128 * 1024 * 1024)
#define WARMUP_ITER 5
#define TEST_ITER 10

typedef void (*test_func_t)(void *src, void *dst, uint64_t size);

static inline void flush_cache(void *ptr, size_t size) {
    __builtin___clear_cache(ptr, (char*)ptr + size);
}

static double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

void test_ldp_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    float *src = (float *)src_ptr;
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

void test_stp_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    float *dst = (float *)dst_ptr;
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

void test_ldp_stp(void *src_ptr, void *dst_ptr, uint64_t size) {
    float *src = (float *)src_ptr;
    float *dst = (float *)dst_ptr;
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

void test_ld1w_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
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

void test_st1w_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *dst = (uint8_t *)dst_ptr;
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

void test_ld1w_st1w(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
    uint8_t *dst = (uint8_t *)dst_ptr;
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

void test_ld1d_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
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

void test_st1d_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *dst = (uint8_t *)dst_ptr;
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

void test_ld1d_st1d(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
    uint8_t *dst = (uint8_t *)dst_ptr;
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

void test_ld1b_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
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

void test_st1b_only(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *dst = (uint8_t *)dst_ptr;
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

void test_ld1b_st1b(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
    uint8_t *dst = (uint8_t *)dst_ptr;
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

void test_ld2w_st2w(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
    uint8_t *dst = (uint8_t *)dst_ptr;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.s\n"
            "ld2w {z0.s, z1.s}, p0/z, [%[s]]\n"
            "ld2w {z2.s, z3.s}, p0/z, [%[s], #2, MUL VL]\n"
            "ld2w {z4.s, z5.s}, p0/z, [%[s], #4, MUL VL]\n"
            "ld2w {z6.s, z7.s}, p0/z, [%[s], #6, MUL VL]\n"
            "st2w {z0.s, z1.s}, p0, [%[d]]\n"
            "st2w {z2.s, z3.s}, p0, [%[d], #2, MUL VL]\n"
            "st2w {z4.s, z5.s}, p0, [%[d], #4, MUL VL]\n"
            "st2w {z6.s, z7.s}, p0, [%[d], #6, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

void test_ld4w_st4w(void *src_ptr, void *dst_ptr, uint64_t size) {
    uint8_t *src = (uint8_t *)src_ptr;
    uint8_t *dst = (uint8_t *)dst_ptr;
    uint64_t vl = svcntb();
    uint64_t chunk_size = vl * 8;
    uint64_t chunks = size / chunk_size;
    for (uint64_t i = 0; i < chunks; i++) {
        __asm__ volatile (
            "ptrue p0.s\n"
            "ld4w {z0.s, z1.s, z2.s, z3.s}, p0/z, [%[s]]\n"
            "ld4w {z4.s, z5.s, z6.s, z7.s}, p0/z, [%[s], #4, MUL VL]\n"
            "st4w {z0.s, z1.s, z2.s, z3.s}, p0, [%[d]]\n"
            "st4w {z4.s, z5.s, z6.s, z7.s}, p0, [%[d], #4, MUL VL]\n"
            "add %[s], %[s], %[inc]\n"
            "add %[d], %[d], %[inc]\n"
            : [s] "+r" (src), [d] "+r" (dst)
            : [inc] "r" (chunk_size)
            : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory"
        );
    }
}

double run_test(test_func_t func, void *src, void *dst, uint64_t size) {
    struct timespec start, end;
    
    for (int i = 0; i < WARMUP_ITER; i++) {
        func(src, dst, size);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < TEST_ITER; i++) {
        func(src, dst, size);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return time_sec / TEST_ITER;
}

int main(int argc, char *argv[]) {
    uint64_t vl = svcntb();
    printf("================================================\n");
    printf("SVE Bandwidth Benchmark\n");
    printf("================================================\n");
    printf("SVE Vector Length: %lu bytes (%lu bits)\n", vl, vl * 8);
    printf("Buffer Size: %d MB\n", BUFFER_SIZE / (1024 * 1024));
    printf("Warmup Iterations: %d\n", WARMUP_ITER);
    printf("Test Iterations: %d\n\n", TEST_ITER);
    
    void *src = NULL, *dst = NULL;
    
    if (posix_memalign(&src, 64, BUFFER_SIZE) != 0 ||
        posix_memalign(&dst, 64, BUFFER_SIZE) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory\n");
        return 1;
    }
    
    memset(src, 0x55, BUFFER_SIZE);
    memset(dst, 0xAA, BUFFER_SIZE);
    
    printf("%-18s %12s %12s %12s\n", "Test", "GB/s", "Time (ms)", "Data (MB)");
    printf("================================================\n");
    
    struct {
        const char *name;
        test_func_t func;
        int is_read;
        int is_write;
    } tests[] = {
        {"NEON LDP (Read)", test_ldp_only, 1, 0},
        {"NEON STP (Write)", test_stp_only, 0, 1},
        {"NEON LDP+STP (Copy)", test_ldp_stp, 1, 1},
        {"SVE LD1B (Read)", test_ld1b_only, 1, 0},
        {"SVE ST1B (Write)", test_st1b_only, 0, 1},
        {"SVE LD1B+ST1B (Copy)", test_ld1b_st1b, 1, 1},
        {"SVE LD1W (Read)", test_ld1w_only, 1, 0},
        {"SVE ST1W (Write)", test_st1w_only, 0, 1},
        {"SVE LD1W+ST1W (Copy)", test_ld1w_st1w, 1, 1},
        {"SVE LD1D (Read)", test_ld1d_only, 1, 0},
        {"SVE ST1D (Write)", test_st1d_only, 0, 1},
        {"SVE LD1D+ST1D (Copy)", test_ld1d_st1d, 1, 1},
        {"SVE LD2W+ST2W (Copy)", test_ld2w_st2w, 1, 1},
        {"SVE LD4W+ST4W (Copy)", test_ld4w_st4w, 1, 1},
    };
    
    for (int i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
        double time_sec = run_test(tests[i].func, src, dst, BUFFER_SIZE);
        uint64_t data_bytes = BUFFER_SIZE;
        if (tests[i].is_read && tests[i].is_write) {
            data_bytes = BUFFER_SIZE * 2;
        }
        double bw = get_bandwidth(data_bytes, time_sec);
        printf("%-18s %12.2f %12.3f %12.0f\n", 
               tests[i].name, bw, time_sec * 1000, (double)data_bytes / (1024 * 1024));
    }
    
    printf("================================================\n");
    
    free(src);
    free(dst);
    return 0;
}