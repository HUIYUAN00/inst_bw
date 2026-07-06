#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static int matrix_dim = 1024;
static double sparsity = 0.01;
static int warmup_iter = 5;
static int test_iter = 10;
static unsigned int random_seed = 42;

typedef struct {
    double re;
    double im;
} complex_double_t;

static uint64_t *row_ptr = NULL;
static int32_t *col_idx = NULL;
static complex_double_t *values = NULL;
static complex_double_t *vector = NULL;
static complex_double_t *result = NULL;
static complex_double_t *result_ref = NULL;

static uint64_t nnz = 0;

/* COO格式条目：存储 (row, col, value) */
typedef struct {
    uint32_t row;
    uint32_t col;
    complex_double_t value;
} coo_entry_t;

/* 比较器：按 (row, col) 升序排列COO条目，用于CSR转换 */
static int compare_coo(const void *a, const void *b) {
    const coo_entry_t *ca = (const coo_entry_t *)a;
    const coo_entry_t *cb = (const coo_entry_t *)b;
    if (ca->row != cb->row) return (ca->row < cb->row) ? -1 : 1;
    return (ca->col < cb->col) ? -1 : (ca->col > cb->col) ? 1 : 0;
}

static inline double get_mflops(uint64_t flops, double time_sec) {
    return (double)flops / time_sec / 1e6;
}

/* hemv计算：利用Hermitian性质，仅扫描上三角部分(col>=row)，
 * 对角线直接乘，非对角线额外通过共轭对称性累加到 y[col]。
 * conj(a)*x = (a.re*x.re + a.im*x.im) + (a.re*x.im - a.im*x.re)i
 *
 * SVE内嵌汇编实现 (fcmla + zip):
 *   寄存器分配:
 *     x6=i  x7=row_start  x8=row_end  x9=j  x10=unused  x11=VL_doubles
 *     x14=地址计算临时寄存器
 *     q4=vec[i](128bit)
 *     z0=i-bcast  z1=col_idx(64bit)  z2=val(lo)  z3=val(hi)
 *     z4=x[col](lo)  z5=x[i]-bcast  z6=x[col](hi)/temp  z7=result(hi)
 *     z8=col*2  z9=unused  z10=temp  z11=内积高半累加器
 *     z14=temp  z15=内积低半累加器
 *     p0=loop/reduce-p_re  p1=col>=i/reduce-p_im  p2=col>i
 *     p3=zip1(p1,p1)  p4=zip1(p2,p2)  p5=zip2(p1,p1)  p6=zip2(p2,p2)
 *     p7=temp  p8=odd-lanes(循环不变量,归约分离实虚部) */
static void spmv_standard(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    (void)size;
    (void)scalar;

    __asm__ __volatile__ (
        "stp x19, x20, [sp, #-80]!\n\t"
        "stp x21, x22, [sp, #16]\n\t"
        "stp x23, x24, [sp, #32]\n\t"
        "stp x29, x30, [sp, #48]\n\t"
        "str x14, [sp, #64]\n\t"
        "mov x19, %[y]\n\t"
        "mov x20, %[val]\n\t"
        "mov x21, %[vec]\n\t"
        "mov x22, %[rp]\n\t"
        "mov x23, %[ci]\n\t"
        "mov x24, %[dim]\n\t"
        "mov x6, #0\n\t"
        "lsl x7, x24, #1\n\t"
        "rdvl x11, #1\n\t"
        "lsr x11, x11, #3\n\t"
        "1:\n\t"
        "whilelt p1.d, x6, x7\n\t"
        "beq 2f\n\t"
        "mov z1.d, #0\n\t"
        "st1d z1.d, p1, [x19, x6, lsl #3]\n\t"
        "add x6, x6, x11\n\t"
        "b 1b\n\t"
        "2:\n\t"
        "ptrue p7.b\n\t"
        "index z0.d, #0, #1\n\t"
        "and z0.d, z0.d, #1\n\t"
        "cmpne p8.d, p7/z, z0.d, #0\n\t"
        "mov x6, #0\n\t"
        "3:\n\t"
        "cmp x6, x24\n\t"
        "bge 4f\n\t"
        "add x14, x22, x6, lsl #3\n\t"
        "ldr x7, [x14]\n\t"
        "ldr x8, [x14, #8]\n\t"
        "add x14, x21, x6, lsl #4\n\t"
        "ldr q4, [x14]\n\t"
        "mov z5.q, q4\n\t"
        "mov z15.d, #0\n\t"
        "mov z11.d, #0\n\t"
        "mov x9, x7\n\t"
        "5:\n\t"
        "cmp x9, x8\n\t"
        "bge 6f\n\t"
        "add x14, x23, x9, lsl #2\n\t"
        "whilelt p0.d, x9, x8\n\t"
        "ld1sw z1.d, p0/z, [x14]\n\t"
        "add x14, x20, x9, lsl #4\n\t"
        "dup z0.d, x6\n\t"
        "cmpge p1.d, p0/z, z1.d, z0.d\n\t"
        "cmpgt p2.d, p0/z, z1.d, z0.d\n\t"
        "zip1 p3.d, p1.d, p1.d\n\t"
        "zip2 p5.d, p1.d, p1.d\n\t"
        "zip1 p4.d, p2.d, p2.d\n\t"
        "zip2 p6.d, p2.d, p2.d\n\t"
        "ld1d z2.d, p3/z, [x14]\n\t"
        "add x14, x14, x11, lsl #3\n\t"
        "ld1d z3.d, p5/z, [x14]\n\t"
        "lsl z8.d, z1.d, #1\n\t"
        "add x14, x21, #8\n\t"
        "ld1d z4.d, p1/z, [x21, z8.d, lsl #3]\n\t"
        "ld1d z6.d, p1/z, [x14, z8.d, lsl #3]\n\t"
        "zip1 z10.d, z4.d, z6.d\n\t"
        "zip2 z4.d, z4.d, z6.d\n\t"
        "fcmla z15.d, p3/m, z2.d, z10.d, #0\n\t"
        "fcmla z15.d, p3/m, z2.d, z10.d, #90\n\t"
        "fcmla z11.d, p5/m, z3.d, z4.d, #0\n\t"
        "fcmla z11.d, p5/m, z3.d, z4.d, #90\n\t"
        "ld1d z6.d, p2/z, [x19, z8.d, lsl #3]\n\t"
        "add x14, x19, #8\n\t"
        "ld1d z7.d, p2/z, [x14, z8.d, lsl #3]\n\t"
        "zip1 z14.d, z6.d, z7.d\n\t"
        "fcmla z14.d, p4/m, z2.d, z5.d, #0\n\t"
        "fcmla z14.d, p4/m, z2.d, z5.d, #270\n\t"
        "zip2 z10.d, z6.d, z7.d\n\t"
        "fcmla z10.d, p6/m, z3.d, z5.d, #0\n\t"
        "fcmla z10.d, p6/m, z3.d, z5.d, #270\n\t"
        "uzp1 z6.d, z14.d, z10.d\n\t"
        "st1d z6.d, p2, [x19, z8.d, lsl #3]\n\t"
        "uzp2 z7.d, z14.d, z10.d\n\t"
        "st1d z7.d, p2, [x14, z8.d, lsl #3]\n\t"
        "add x9, x9, x11\n\t"
        "b 5b\n\t"
        "6:\n\t"
        "ptrue p7.b\n\t"
        "not p0.b, p7/z, p8.b\n\t"
        "faddv d0, p0, z15.d\n\t"
        "faddv d1, p0, z11.d\n\t"
        "fadd d0, d0, d1\n\t"
        "add x14, x19, x6, lsl #4\n\t"
        "ldr d4, [x14]\n\t"
        "fadd d4, d4, d0\n\t"
        "str d4, [x14]\n\t"
        "mov p7.b, p8.b\n\t"
        "faddv d0, p7, z15.d\n\t"
        "faddv d1, p7, z11.d\n\t"
        "fadd d0, d0, d1\n\t"
        "ldr d4, [x14, #8]\n\t"
        "fadd d4, d4, d0\n\t"
        "str d4, [x14, #8]\n\t"
        "add x6, x6, #1\n\t"
        "b 3b\n\t"
        "4:\n\t"
        "ldr x14, [sp, #64]\n\t"
        "ldp x29, x30, [sp, #48]\n\t"
        "ldp x23, x24, [sp, #32]\n\t"
        "ldp x21, x22, [sp, #16]\n\t"
        "ldp x19, x20, [sp], #80\n\t"
        : /* no outputs */
        : [y] "r"(result_ptr), [val] "r"(values_ptr), [vec] "r"(vector_ptr),
          [rp] "r"(row_ptr), [ci] "r"(col_idx), [dim] "r"(matrix_dim)
        : "x6", "x7", "x8", "x9", "x10", "x11", "x14",
          "x19", "x20", "x21", "x22", "x23", "x24",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
          "memory", "cc"
    );
}

/* hemv校验函数：利用Hermitian性质，仅扫描上三角部分(col>=row)，
 * 对角线直接乘，非对角线额外通过共轭对称性累加到 y[col]。
 * conj(a)*x = (a.re*x.re + a.im*x.im) + (a.re*x.im - a.im*x.re)i */
static void hermitian_spmv_scalar(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    complex_double_t *val = (complex_double_t *)values_ptr;
    complex_double_t *vec = (complex_double_t *)vector_ptr;
    complex_double_t *y = (complex_double_t *)result_ptr;
    
    for (uint64_t i = 0; i < matrix_dim; i++) {
        y[i].re = 0.0;
        y[i].im = 0.0;
    }
    
    for (uint64_t i = 0; i < matrix_dim; i++) {
        for (uint64_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            int32_t col = col_idx[j];
            if (col < (int32_t)i) continue;
            
            complex_double_t a = val[j];
            complex_double_t xj = vec[col];
            
            y[i].re += a.re * xj.re - a.im * xj.im;
            y[i].im += a.re * xj.im + a.im * xj.re;
            
            if (col != (int32_t)i) {
                complex_double_t xi = vec[i];
                y[col].re += a.re * xi.re + a.im * xi.im;
                y[col].im += a.re * xi.im - a.im * xi.re;
            }
        }
    }
}

typedef struct {
    const char *name;
    const char *category;
    void (*func)(void *result, void *values, void *vector, uint64_t size, double scalar);
} test_item_t;

static test_item_t test_registry[] = {
    {"Complex SpMV", "SpMV", spmv_standard},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -n, --dim <N>           Matrix dimension N (N x N) (default: 1024)\n");
    printf("  -s, --sparsity <ratio>  Sparsity ratio 0.0-1.0 (default: 0.01)\n");
    printf("  -r, --random-seed <N>   Random seed (default: 42)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("\nNote: Tests Hermitian matrix SpMV with full CSR storage\n");
    printf("      Diagonal elements are real (imaginary part = 0)\n");
    printf("      Verification uses hemv (upper triangle + conjugate symmetry)\n");
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

static int verify_spmv_result(void *result_ptr, void *ref_ptr) {
    complex_double_t *y = (complex_double_t *)result_ptr;
    complex_double_t *y_ref = (complex_double_t *)ref_ptr;
    int errors = 0;
    
    for (uint64_t i = 0; i < matrix_dim && errors < 5; i++) {
        double re_diff = fabs(y[i].re - y_ref[i].re);
        double im_diff = fabs(y[i].im - y_ref[i].im);
        if (re_diff > 1e-9 || im_diff > 1e-9) {
            if (errors == 0) fprintf(stderr, "SpMV verify FAILED:\n");
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
    struct timeval start, end;
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    for (int i = 0; i < warmup_iter; i++) {
        test->func(result, values, vector, nnz, 1.0);
    }
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    gettimeofday(&start, NULL);
    for (int i = 0; i < test_iter; i++) {
        test->func(result, values, vector, nnz, 1.0);
    }
    gettimeofday(&end, NULL);
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
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
        if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--dim") == 0) {
            if (i + 1 < argc) matrix_dim = atoi(argv[++i]);
            if (matrix_dim < 1) matrix_dim = 1024;
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
            if (warmup_iter < 0) warmup_iter = 5;
            continue;
        }
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            if (i + 1 < argc) test_iter = atoi(argv[++i]);
            if (test_iter < 1) test_iter = 10;
            continue;
        }
        run_all = 0;
        num_specs++;
    }
    
    if (!run_all && num_specs > 0) {
        specs = &argv[argc - num_specs];
    }
    
#ifdef USE_MPI
    MPI_Bcast(&matrix_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    
    uint64_t total_elements = (uint64_t)matrix_dim * matrix_dim;
    nnz = (uint64_t)(total_elements * sparsity);
    if (nnz < (uint64_t)matrix_dim) nnz = matrix_dim;
    
    if (rank == 0) {
        printf("================================================================================\n");
#ifdef USE_MPI
        printf("Hermitian SpMV Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("Hermitian SpMV Benchmark\n");
#endif
        printf("================================================================================\n");
        printf("Matrix Dimension: %d x %d\n", matrix_dim, matrix_dim);
        printf("Sparsity: %.4f (%.2f%%)\n", sparsity, sparsity * 100);
        printf("Total Elements: %lu\n", total_elements);
        printf("NNZ: %lu\n", nnz);
        printf("Avg NNZ per Row: %.2f\n", (double)nnz / matrix_dim);
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Random Seed: %u\n", random_seed);
        printf("\n");
    }
    
    row_ptr = (uint64_t *)malloc((matrix_dim + 1) * sizeof(uint64_t));
    col_idx = (int32_t *)malloc(nnz * sizeof(int32_t));
    values = (complex_double_t *)malloc(nnz * sizeof(complex_double_t));
    vector = (complex_double_t *)malloc(matrix_dim * sizeof(complex_double_t));
    result = (complex_double_t *)malloc(matrix_dim * sizeof(complex_double_t));
    result_ref = (complex_double_t *)malloc(matrix_dim * sizeof(complex_double_t));
    
    if (!row_ptr || !col_idx || !values || !vector || !result || !result_ref) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate memory\n");
#endif
        return 1;
    }
    
    srand(random_seed);
    
    for (uint64_t i = 0; i < matrix_dim; i++) {
        result[i].re = (double)rand() / RAND_MAX;
        result[i].im = (double)rand() / RAND_MAX;
        result_ref[i].re = (double)rand() / RAND_MAX;
        result_ref[i].im = (double)rand() / RAND_MAX;
    }
    
    /* 生成随机输入向量 */
    for (uint64_t i = 0; i < matrix_dim; i++) {
        vector[i].re = (double)rand() / RAND_MAX;
        vector[i].im = (double)rand() / RAND_MAX;
    }
    
    /* ===== 阶段1：通过随机采样构建COO格式矩阵 ===== */
    /* 分配COO数组及用于全矩阵位置去重的bitmap */
    coo_entry_t *coo = (coo_entry_t *)malloc(nnz * sizeof(coo_entry_t));
    uint64_t *coverage = (uint64_t *)calloc((total_elements / 64) + 2, sizeof(uint64_t));
    uint64_t unique_count = 0;
    int attempts = 0;
    int max_attempts = nnz * 20;
    
    /* 阶段1a：随机采样 + bitmap去重
     * 将全矩阵展平为一维索引 [0, total_elements)，用bitmap保证每个位置最多采样一次。
     * 映射回 (row, col)：row = idx / N, col = idx % N。
     * 赋值时对角线虚部为0，非对角线为随机复数。
     * 对于 Hermitian 矩阵，同时生成 (row, col) 和 (col, row) 共轭对。 */
    while (unique_count < nnz && attempts < max_attempts) {
        uint64_t rand_val = ((uint64_t)rand() << 32) | (uint64_t)rand();
        uint64_t idx = rand_val % total_elements;
        uint64_t bucket = idx / 64;
        uint64_t bit = idx % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            uint64_t row = idx / matrix_dim;
            uint64_t col = idx % matrix_dim;
            
            if (row == col) {
                coverage[bucket] |= (1ULL << bit);
                coo[unique_count].row = (uint32_t)row;
                coo[unique_count].col = (uint32_t)col;
                coo[unique_count].value.re = (double)rand() / RAND_MAX;
                coo[unique_count].value.im = 0.0;
                unique_count++;
            } else {
                uint64_t sym_idx = col * matrix_dim + row;
                uint64_t sym_bucket = sym_idx / 64;
                uint64_t sym_bit = sym_idx % 64;
                if (!(coverage[sym_bucket] & (1ULL << sym_bit))) {
                    coverage[bucket] |= (1ULL << bit);
                    coverage[sym_bucket] |= (1ULL << sym_bit);
                    double re = (double)rand() / RAND_MAX;
                    double im = (double)rand() / RAND_MAX;
                    coo[unique_count].row = (uint32_t)row;
                    coo[unique_count].col = (uint32_t)col;
                    coo[unique_count].value.re = re;
                    coo[unique_count].value.im = im;
                    unique_count++;
                    if (unique_count < nnz) {
                        coo[unique_count].row = (uint32_t)col;
                        coo[unique_count].col = (uint32_t)row;
                        coo[unique_count].value.re = re;
                        coo[unique_count].value.im = -im;
                        unique_count++;
                    }
                }
            }
        }
        attempts++;
    }
    
    /* 阶段1b：顺序兜底填充——若随机采样未达到 nnz，
     * 顺序扫描剩余未覆盖位置补足数量。
     * 对于 Hermitian 矩阵，同时生成 (row, col) 和 (col, row) 共轭对。 */
    for (uint64_t idx = 0; idx < total_elements && unique_count < nnz; idx++) {
        uint64_t bucket = idx / 64;
        uint64_t bit = idx % 64;
        if (!(coverage[bucket] & (1ULL << bit))) {
            uint64_t row = idx / matrix_dim;
            uint64_t col = idx % matrix_dim;
            
            if (row == col) {
                coverage[bucket] |= (1ULL << bit);
                coo[unique_count].row = (uint32_t)row;
                coo[unique_count].col = (uint32_t)col;
                coo[unique_count].value.re = (double)rand() / RAND_MAX;
                coo[unique_count].value.im = 0.0;
                unique_count++;
            } else {
                uint64_t sym_idx = col * matrix_dim + row;
                uint64_t sym_bucket = sym_idx / 64;
                uint64_t sym_bit = sym_idx % 64;
                if (!(coverage[sym_bucket] & (1ULL << sym_bit))) {
                    coverage[bucket] |= (1ULL << bit);
                    coverage[sym_bucket] |= (1ULL << sym_bit);
                    double re = (double)rand() / RAND_MAX;
                    double im = (double)rand() / RAND_MAX;
                    coo[unique_count].row = (uint32_t)row;
                    coo[unique_count].col = (uint32_t)col;
                    coo[unique_count].value.re = re;
                    coo[unique_count].value.im = im;
                    unique_count++;
                    if (unique_count < nnz) {
                        coo[unique_count].row = (uint32_t)col;
                        coo[unique_count].col = (uint32_t)row;
                        coo[unique_count].value.re = re;
                        coo[unique_count].value.im = -im;
                        unique_count++;
                    }
                }
            }
        }
    }
    
    free(coverage);
    
    /* ===== 阶段2：按 (row, col) 排序COO条目，为CSR转换做准备 ===== */
    qsort(coo, nnz, sizeof(coo_entry_t), compare_coo);
    
    /* ===== 阶段3：将排序后的COO转换为CSR格式 ===== */
    /* 从排序后的COO条目构建 row_ptr、col_idx、values 数组。
     * row_ptr[i] 记录第i行在 col_idx/values 中的起始偏移。 */
    row_ptr[0] = 0;
    uint64_t current_row = 0;
    
    for (uint64_t i = 0; i < nnz; i++) {
        col_idx[i] = (int32_t)coo[i].col;
        values[i] = coo[i].value;
        
        /* 为跳过的空行填充 row_ptr */
        while (current_row < coo[i].row) {
            row_ptr[current_row + 1] = i;
            current_row++;
        }
    }
    
    /* 为尾部空行填充 row_ptr，直到 matrix_dim */
    while (current_row < matrix_dim) {
        row_ptr[current_row + 1] = nnz;
        current_row++;
    }
    
    free(coo);
    
    /* ===== 阶段4：一次性校验 ===== */
    hermitian_spmv_scalar(result_ref, values, vector, nnz, 1.0);
    spmv_standard(result, values, vector, nnz, 1.0);
    
    int verify_errors = verify_spmv_result(result, result_ref);
    if (rank == 0) {
        if (verify_errors > 0) {
            printf("Verification FAILED (%d errors). Skipping benchmark.\n", verify_errors);
        } else {
            printf("Verification PASS\n");
        }
        printf("\n");
    }
    
    if (verify_errors > 0) {
        free(row_ptr); free(col_idx); free(values);
        free(vector); free(result); free(result_ref);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
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
        uint64_t flops = nnz * 8;
        
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
        
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        
        if (rank == 0) {
#ifdef USE_MPI
            printf("%-38s %14s %12.2f %12.3f %12.2f",
                   test->name, test->category, mflops, time_sec * 1000, total_mflops);
#else
            printf("%-38s %14s %12.2f %12.3f",
                   test->name, test->category, mflops, time_sec * 1000);
#endif
            printf("  PASS\n");
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