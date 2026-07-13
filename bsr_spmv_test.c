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
static uint64_t matrix_size = 256;  /* 矩阵某维度上 block 的个数 (实际矩阵维度 = matrix_size * block_size) */
static int block_size = 4;
static double sparsity = 0.1;
static unsigned int random_seed = 42;
static int print_all_ranks = 0;

typedef struct {
    float re;
    float im;
} complex_float_t;

static uint64_t nnz_blocks = 0;
static uint64_t *row_ptr = NULL;
static int32_t *col_idx = NULL;
static complex_float_t *values = NULL;
static complex_float_t *vector = NULL;
static complex_float_t *result = NULL;
static complex_float_t *result_ref = NULL;

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
 * BSR Hermitian SpMV: y = A*x (complex float)
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
    complex_float_t *val = (complex_float_t *)values_ptr;
    complex_float_t *vec = (complex_float_t *)vector_ptr;
    complex_float_t *y = (complex_float_t *)result_ptr;
    int r = block_size;
    uint64_t nbr = matrix_size;

    for (uint64_t i = 0; i < matrix_size * r; i++) {
        y[i].re = 0.0f;
        y[i].im = 0.0f;
    }

    for (uint64_t I = 0; I < nbr; I++) {
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            int32_t J = col_idx[j];
            complex_float_t *block = &val[j * r * r];

            for (int i = 0; i < r; i++) {
                for (int k = 0; k < r; k++) {
                    complex_float_t a = block[i * r + k];
                    complex_float_t xj = vec[J * r + k];
                    y[I * r + i].re += a.re * xj.re - a.im * xj.im;
                    y[I * r + i].im += a.re * xj.im + a.im * xj.re;
                }
            }

            if (I < J) {
                complex_float_t *xi = &vec[I * r];
                for (int k = 0; k < r; k++) {
                    for (int i = 0; i < r; i++) {
                        complex_float_t a = block[i * r + k];
                        y[J * r + k].re += a.re * xi[i].re + a.im * xi[i].im;
                        y[J * r + k].im += a.re * xi[i].im - a.im * xi[i].re;
                    }
                }
            }
        }
    }
}

/*
 * BSR Hermitian SpMV (SVE向量化): y = A*x, A为Hermitian矩阵
 *
 * 矩阵存储: 仅存上三角块(I<=J),下三角通过共轭对称性推导。
 *   BSR三数组: row_ptr[nbr+1], col_idx[nnz], values[nnz*r*r]
 *   块内 r×r 元素按行主序存储,每个元素为 complex_float_t {re, im}
 *
 * 内存布局约定(关键):
 *   complex_float_t 数组按 float 视角读取时,布局为 [re,im,re,im,...] 交错。
 *   即 block[bi][bk] 在 float 视角下占 2 个连续 lane: 偶数lane=实部, 奇数lane=虚部。
 *   SVE 向量化利用此交错布局,用 svcmla (FCMLA指令) 在单条指令内完成复数乘加。
 *
 * svcmla (FCMLA) 旋转参数语义(对交错复数对 [a_re,a_im] × [b_re,b_im]):
 *   #0:   acc_re += a_re*b_re - a_im*b_im,  acc_im += a_re*b_im + a_im*b_re   (= a*b)
 *   #90:  acc_re += -a_re*b_im - a_im*b_re, acc_im += a_re*b_re - a_im*b_im    (= a*b*i)
 *   #270: acc_re += a_re*b_im + a_im*b_re,  acc_im += -a_re*b_re + a_im*b_im   (= -a*b*i)
 *   组合 #0 + #90 实现完整复数乘法 a*b (两步,因FCMLA每次只贡献一半交叉项)。
 *   组合 #0 + #270 实现 conj(a)*b (交换虚部符号,用于共轭转置贡献)。
 *
 * svcmla 逐 lane 对操作: 偶数lane累积实部, 奇数lane累积虚部。
 *   #0 + #90 后, zacc = [re_acc, im_acc, re_acc, im_acc, ...]
 *   svuzp1 提取偶数lane [re, re, ...], svuzp2 提取奇数lane [im, im, ...]
 *   svaddv 分别归约实部/虚部,得到标量 sum_re / sum_im。
 *
 * 复数标量广播方法:
 *   方式1: svzip1_f32(svdup_f32(xre), svdup_f32(xim)) → [xre,xim,xre,xim,...] (2条指令)
 *   方式2: svreinterpret_f32_u64(svdup_u64(*(uint64_t*)&x)) → [xre,xim,xre,xim,...] (1条dup+1条reinterpret)
 *   方式2 将复数 {re,im} 视为 64-bit 打包值一次性广播,更简洁。
 *
 * 遍历策略: 单趟行主序,外层循环块行 I=0..nbr-1,内层循环该行所有非零块。
 *
 *   对角块(J==I): 块本身为Hermitian,仅读块内上三角(bi<=bk)数据
 *     循环结构: for(bi) { while(off) { 内积(pg_upper) + 外积(pg_strict) } 归约 }
 *     即 r 维循环(bi)在外, 向量化 while 循环在内, block数据每轮加载一次,内积外积融合。
 *     - 对角线上(bi<=bk)内积: y[I*r+bi] += a * x[I*r+bk]
 *     - 对角线下(bi<bk)外积: y[I*r+bk] += conj(a) * x[I*r+bi]
 *       (利用 block[bk][bi]=conj(block[bi][bk]),数据从上三角读取,贡献到下三角)
 *
 *   上三角块(J>I): 完整块乘 + 共轭转置贡献
 *     循环结构: while(off) { for(i) { 内积 / 外积 } }
 *     即向量化 while 循环在外(遍历k方向的向量chunk), r 维循环(i)在最内层。
 *     x/y 向量chunk 在外层加载一次,内层 r 行复用,减少访存次数。
 *     - 内积: y[I*r+i] += a * x[J*r+k]     (非对角块数据乘向量)
 *     - 外积: y[J*r+k] += conj(a) * x[I*r+i] (共轭转置乘向量,下三角贡献)
 */
static void spmv_bsr_herm_sve(void *result_ptr, void *values_ptr, void *vector_ptr, uint64_t size, double scalar) {
    (void)size;
    (void)scalar;
    complex_float_t *val = (complex_float_t *)values_ptr;
    complex_float_t *vec = (complex_float_t *)vector_ptr;
    complex_float_t *y = (complex_float_t *)result_ptr;
    int r = block_size;
    uint64_t nbr = matrix_size;
    uint64_t vl_floats = svcntw();  /* SVE向量寄存器可容纳的float32数 */

    /* 清零结果向量 */
    for (uint64_t i = 0; i < matrix_size * r; i++) {
        y[i].re = 0.0f;
        y[i].im = 0.0f;
    }

    /* ===== 单趟遍历: 行主序外层循环,逐块识别对角/上三角 ===== */
    for (uint64_t I = 0; I < nbr; I++) {
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            int32_t J = col_idx[j];
            complex_float_t *block = &val[j * r * r];

            if ((uint64_t)J == I) {
                /* ============================================================
                 * 对角块(I==J): 块内 Hermitian,仅读上三角(bi<=bk)
                 *
                 * 块内布局 (r=4 示例),float视角,[]标记是否在读取范围内:
                 *         bk=0      bk=1      bk=2      bk=3
                 *   bi=0  [a00r a00i][a01r a01i][a02r a02i][a03r a03i]  ← 全部上三角
                 *   bi=1  [a10r a10i][a11r a11i][a12r a12i][a13r a13i]  ← a10* 已跳过,仅 a11+ 上三角
                 *   bi=2  [a20r a20i][a21r a21i][a22r a22i][a23r a23i]  ← a20*,a21* 跳过,仅 a22+ 上三角
                 *   bi=3  [a30r a30i][a31r a31i][a32r a32i][a33r a33i]  ← 仅 a33 上三角
                 *
                 * Hermitian性质: block[bk][bi] = conj(block[bi][bk])
                 *   对角线(bi==bk): block[bi][bi] 为实数(虚部=0)
                 *   下三角(bi>bk): 可由上三角共轭推导,无需存储
                 *
                 * 循环结构: r维循环(bi)为外层, 向量化while循环(off)为内层。
                 *   每轮 off 迭代加载 block第bi行(za) 和 x向量(zx) 各一次,
                 *   内积外积融合处理,block数据不重复读取。
                 *
                 *   (1) 内积(bk>=bi): y[bi] += block[bi][bk] * x[bk]
                 *       谓词 pg_upper 选择 bk>=bi 的复数对 (float偏移 >= 2*bi)
                 *       svcmla #0 + #90 完成复数乘法 a*x, 累加到 zacc
                 *       循环结束后统一归约 zacc → y[bi]
                 *
                 *   (2) 外积(bk>bi): y[bk] += conj(block[bi][bk]) * x[bi]
                 *       (利用上三角数据 block[bi][bk] 计算下三角 block[bk][bi] 的共轭贡献)
                 *       谓词 pg_strict 选择 bk>bi 的复数对 (float偏移 >= 2*(bi+1))
                 *       x[bi] 视为 64-bit (re,im 打包), svdup_u64 广播为 [xre,xim,xre,xim,...]
                 *       svcmla #0 + #270 计算 conj(a)*x:
                 *         #0:   [re += ar*xre,    im += ar*xim]
                 *         #270: [re += ai*xim,    im += -ai*xre]
                 *         合计: re += ar*xre+ai*xim, im += ar*xim-ai*xre ✓
                 *       原地读改写 zy, 仅写回 pg_strict lane
                 *
                 * 注: 外积用 FCMLA 而非 svmla+uzp1/uzp2,因 FCMLA 在原始交错布局上
                 *     直接运算,谓词 pg_strict 能正确对应到原始 lane 位置。
                 * ============================================================ */
                complex_float_t *xi = &vec[I * r];
                complex_float_t *yi = &y[I * r];
                uint64_t total = (uint64_t)r * 2;  /* 块一行的 float 总数 (r个复数 × 2) */

                for (int bi = 0; bi < r; bi++) {
                    float *br = (float *)&block[bi * r];  /* block第bi行, float视角 */
                    float *xr = (float *)xi;               /* x向量, float视角 */
                    float *yr = (float *)yi;               /* y向量, float视角 */

                    /* 外积广播向量: x[bi] 视为 64-bit (re,im 打包), dup 广播到整个寄存器
                     * svdup_u64 → [re,im,re,im,...] (64-bit lane = 1 个复数)
                     * svreinterpret_f32 转为 float32 视角,直接用于 FCMLA */
                    svfloat32_t zx_bcast = svreinterpret_f32_u64(svdup_u64(*(uint64_t *)&xi[bi]));

                    /* 内积累加器: 跨 while 迭代累加,循环结束后统一归约 */
                    svfloat32_t zacc = svdup_f32(0.0f);

                    uint64_t off = 0;
                    while (off < total) {
                        uint64_t cnt = (total - off < vl_floats) ? (total - off) : vl_floats;
                        svbool_t pgall = svptrue_b32();
                        svbool_t pg = svcmplt_u32(pgall, svindex_u32(0, 1), svdup_u32(cnt));
                        /* pg_upper: 绝对偏移 >= 2*bi → bk>=bi; pg_strict: >= 2*(bi+1) → bk>bi */
                        svbool_t pg_upper = svcmpge_u32(pg, svindex_u32(off, 1), svdup_u32(2 * bi));
                        svbool_t pg_strict = svcmpge_u32(pg, svindex_u32(off, 1), svdup_u32(2 * (bi + 1)));

                        /* block 第bi行数据仅加载一次,内积外积共用 */
                        svfloat32_t za = svld1_f32(pg, br + off);
                        svfloat32_t zx = svld1_f32(pg, xr + off);

                        /* (1) 内积(bk>=bi): zacc += a * x[bk], 累加到 zacc */
                        zacc = svcmla_f32_m(pg_upper, zacc, za, zx, 0);   /* 实部交叉项 */
                        zacc = svcmla_f32_m(pg_upper, zacc, za, zx, 90);  /* 虚部交叉项 */

                        /* (2) 外积(bk>bi): y[bk] += conj(a) * x[bi], 原地读改写 zy */
                        svfloat32_t zy = svld1_f32(pg, yr + off);
                        zy = svcmla_f32_m(pg_strict, zy, za, zx_bcast, 0);    /* re += ar*xre, im += ar*xim */
                        zy = svcmla_f32_m(pg_strict, zy, za, zx_bcast, 270);  /* re += ai*xim, im += -ai*xre */
                        svst1_f32(pg_strict, yr + off, zy);

                        off += cnt;
                    }

                    /* 内积累加器归约: zacc=[re,im,re,im,...] → sum_re, sum_im → y[bi] */
                    svbool_t pgall = svptrue_b32();
                    svfloat32_t zzero = svdup_f32(0.0f);
                    svfloat32_t zre = svuzp1_f32(zacc, zzero);  /* 提取偶数lane(实部累加) */
                    svfloat32_t zim = svuzp2_f32(zacc, zzero);  /* 提取奇数lane(虚部累加) */
                    yi[bi].re += svaddv_f32(pgall, zre);
                    yi[bi].im += svaddv_f32(pgall, zim);
                }
            } else {
                /* ============================================================
                 * 上三角块(I<J): 完整块内积 + 共轭转置外积
                 *
                 * (1) 内积: y[I*r+i] += block[i][k] * x[J*r+k],  i,k = 0..r-1
                 *     块数据完整乘以 J 块列对应的向量段。
                 * (2) 外积: y[J*r+k] += conj(block[i][k]) * x[I*r+i],  i,k = 0..r-1
                 *     共轭转置贡献,补全下三角 A[J][I] = conj(A[I][J]) 的乘法。
                 *
                 * 循环结构: 向量化while循环(off)为外层, r维循环(i)为最内层。
                 *   x/y 向量chunk 在外层加载一次,内层 r 行复用,减少访存次数。
                 *   block 每行数据在内层循环中按需加载,各行独立计算。
                 * ============================================================ */

                /* ---- (1) 内积: y[I*r+i] += a * x[J*r+k] ----
                 * 向量化循环为外层(遍历k方向的向量chunk), r维循环为内层(遍历块各行i)
                 * zx(x向量chunk) 所有行共享, 每行独立归约后直接累加到 y
                 * svcmla #0 + #90 完成复数乘法 a*x, svuzp1/svuzp2 分离, svaddv 归约 */
                float *xr = (float *)&vec[J * r];
                uint64_t total = (uint64_t)r * 2;
                svbool_t pgall = svptrue_b32();
                svfloat32_t zzero = svdup_f32(0.0f);

                uint64_t off = 0;
                while (off < total) {
                    uint64_t cnt = (total - off < vl_floats) ? (total - off) : vl_floats;
                    svbool_t pg = svcmplt_u32(pgall, svindex_u32(0, 1), svdup_u32(cnt));
                    svfloat32_t zx = svld1_f32(pg, xr + off);  /* x向量chunk,所有行共享 */

                    for (int i = 0; i < r; i++) {
                        float *br = (float *)&block[i * r];
                        svfloat32_t za = svld1_f32(pg, br + off);
                        svfloat32_t zacc = svdup_f32(0.0f);
                        zacc = svcmla_f32_m(pg, zacc, za, zx, 0);   /* 实部交叉项 */
                        zacc = svcmla_f32_m(pg, zacc, za, zx, 90);  /* 虚部交叉项 */
                        svfloat32_t zre = svuzp1_f32(zacc, zzero);  /* 提取偶数lane(实部) */
                        svfloat32_t zim = svuzp2_f32(zacc, zzero);  /* 提取奇数lane(虚部) */
                        y[I * r + i].re += svaddv_f32(pgall, zre);
                        y[I * r + i].im += svaddv_f32(pgall, zim);
                    }
                    off += cnt;
                }

                /* ---- (2) 外积: y[J*r+k] += conj(a) * x[I*r+i] ----
                 * 向量化循环为外层(遍历k方向的向量chunk), r维循环为内层(遍历块各行i)
                 * zy(y向量chunk) 在外层加载一次,内层 r 行共享累加,最后统一写回
                 * 每行贡献 conj(block[i][k]) * x[i]
                 *
                 * conj(a)*x 实现 (FCMLA #0 + #270):
                 *   x[i] 视为 64-bit (re,im 打包), svdup_u64 广播为 [xre,xim,xre,xim,...]
                 *   FCMLA #0:   [re += ar*xre,    im += ar*xim]
                 *   FCMLA #270: [re += ai*xim,    im += -ai*xre]
                 *   合计: re += ar*xre+ai*xim, im += ar*xim-ai*xre ✓ = conj(a)*x */
                complex_float_t *xi = &vec[I * r];
                float *yr = (float *)&y[J * r];

                off = 0;
                while (off < total) {
                    uint64_t cnt = (total - off < vl_floats) ? (total - off) : vl_floats;
                    svbool_t pg = svcmplt_u32(pgall, svindex_u32(0, 1), svdup_u32(cnt));
                    svfloat32_t zy = svld1_f32(pg, yr + off);  /* y向量chunk,所有行共享 */

                    for (int i = 0; i < r; i++) {
                        float *br = (float *)&block[i * r];
                        svfloat32_t za = svld1_f32(pg, br + off);
                        /* x[i] 视为 64-bit (re,im 打包), dup 广播到整个寄存器 */
                        svfloat32_t zx_bcast = svreinterpret_f32_u64(svdup_u64(*(uint64_t *)&xi[i]));
                        zy = svcmla_f32_m(pg, zy, za, zx_bcast, 0);    /* re += ar*xre, im += ar*xim */
                        zy = svcmla_f32_m(pg, zy, za, zx_bcast, 270);  /* re += ai*xim, im += -ai*xre */
                    }
                    svst1_f32(pg, yr + off, zy);  /* 统一写回,所有行贡献已累加 */
                    off += cnt;
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
    printf("  -M, --matrix-size <N>   Number of blocks per dimension (default: 256)\n");
    printf("  -b, --block-size <r>    Block size r (r x r) (default: 4)\n");
    printf("  -s, --sparsity <ratio>  Block sparsity ratio 0.0-1.0 (default: 0.1)\n");
    printf("  -r, --random-seed <N>   Random seed (default: 42)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -p, --print-all         Print all ranks' results (MPI only)\n");
    printf("\nNote: Tests Hermitian BSR SpMV (complex float) with upper triangle block storage\n");
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
    complex_float_t *y = (complex_float_t *)result_ptr;
    complex_float_t *y_ref = (complex_float_t *)ref_ptr;
    int errors = 0;

    for (uint64_t i = 0; i < matrix_size * block_size && errors < 5; i++) {
        float re_diff = fabsf(y[i].re - y_ref[i].re);
        float im_diff = fabsf(y[i].im - y_ref[i].im);
        float scale = fmaxf(fmaxf(fabsf(y_ref[i].re), fabsf(y_ref[i].im)), 1.0f);
        if (re_diff > 1e-4 * scale || im_diff > 1e-4 * scale) {
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

    if (matrix_size < 1) {
        if (rank == 0) {
            fprintf(stderr, "Error: matrix_size (%lu) must be >= 1\n",
                    matrix_size);
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

    uint64_t num_block_rows = matrix_size;
    uint64_t matrix_dim = matrix_size * block_size;
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
        printf("Matrix Size: %lu x %lu (blocks: %lu x %lu)\n", matrix_dim, matrix_dim, matrix_size, matrix_size);
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
        posix_memalign((void**)&values, 64, nnz_elements * sizeof(complex_float_t)) != 0 ||
        posix_memalign((void**)&vector, 64, matrix_dim * sizeof(complex_float_t)) != 0 ||
        posix_memalign((void**)&result, 64, matrix_dim * sizeof(complex_float_t)) != 0 ||
        posix_memalign((void**)&result_ref, 64, matrix_dim * sizeof(complex_float_t)) != 0) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate aligned memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate aligned memory\n");
#endif
        return 1;
    }

    srand(random_seed);

    for (uint64_t i = 0; i < matrix_dim; i++) {
        vector[i].re = (float)rand() / RAND_MAX;
        vector[i].im = (float)rand() / RAND_MAX;
    }

    /* ===== 阶段1：在上三角块网格中随机采样非零块位置 =====
     *
     * 矩阵被划分为 nbr×nbr 的块网格(nbr = matrix_size),每块 r×r 元素。
     * Hermitian 矩阵只需存储上三角块(I<=J),下三角通过共轭对称性推导。
     * 上三角块总数 = nbr*(nbr+1)/2,从中按 sparsity 比例采样 nnz_blocks 个。
     *
     * 采样策略: bitmap 去重,保证每个块位置最多采一次。
     * 块位置 (I,J) 映射为一维索引: idx = I*nbr - I*(I+1)/2 + J
     *   (仅对 I<=J 有效,即上三角的紧凑编号)
     *
     * 数据结构: coords[] 数组暂存 (block_row, block_col) 对,采样后排序转 CSR。
     */
    block_coord_t *coords = (block_coord_t *)malloc(nnz_blocks * sizeof(block_coord_t));
    uint64_t *coverage = (uint64_t *)calloc((total_upper_blocks / 64) + 2, sizeof(uint64_t));
    uint64_t unique_count = 0;
    int attempts = 0;
    int max_attempts = nnz_blocks * 20;

    /* 阶段1a: 随机采样 */
    while (unique_count < nnz_blocks && attempts < max_attempts) {
        uint64_t rand_val = ((uint64_t)rand() << 32) | (uint64_t)rand();
        uint64_t I = rand_val % num_block_rows;
        uint64_t J = rand_val % num_block_rows;
        if (I > J) { uint64_t t = I; I = J; J = t; }  /* 强制 I<=J,仅取上三角 */
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

    /* 阶段1b: 顺序兜底——若随机采样不足,顺序扫描上三角补齐 */
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

    /* ===== 阶段2: 按 (block_row, block_col) 升序排序 =====
     *
     * 排序后 coords[] 按块行主序排列,为 CSR 转换做准备。
     */
    qsort(coords, nnz_blocks, sizeof(block_coord_t), compare_coord);

    /* ===== 阶段3: 从排序后的 coords 构建 BSR 的三个核心数组 =====
     *
     * 最终 BSR 数据结构(三个数组):
     *
     *   row_ptr[nbr+1]: 块行指针
     *     row_ptr[I] 到 row_ptr[I+1]-1 是第 I 个块行的非零块在 col_idx/values 中的连续偏移
     *     长度 = num_block_rows + 1, row_ptr[0]=0, row_ptr[nbr]=nnz_blocks
     *
     *   col_idx[nnz_blocks]: 块列索引
     *     col_idx[j] = 第 j 个非零块所在的块列号 J
     *     对应原始矩阵的列范围 [J*r, (J+1)*r)
     *
     *   values[nnz_blocks * r * r]: 块数据,行主序存储
     *     第 j 个块从 values[j*r*r] 开始,连续 r*r 个 complex_float_t
     *     块内元素 (bi,bk) 存于 values[j*r*r + bi*r + bk]
     *     对应原始矩阵 A[I*r+bi][J*r+bk]
     *
     * 示意(nbr=3, r=2, nnz_blocks=4):
     *   块网格(上三角):     BSR 数组:
     *     I=0: [0,0][0,2]    row_ptr = [0, 2, 3, 4]
     *     I=1: [1,1]         col_idx = [0, 2, 1, 2]
     *     I=2: [2,2]         values  = [B00|B02|B11|B22]  每块 4 个元素
     */
    row_ptr[0] = 0;
    uint64_t current_row = 0;

    for (uint64_t i = 0; i < nnz_blocks; i++) {
        col_idx[i] = (int32_t)coords[i].col;
        /* 为跳过的空块行填充 row_ptr */
        while (current_row < coords[i].row) {
            row_ptr[current_row + 1] = i;
            current_row++;
        }
    }
    /* 为尾部空块行填充 row_ptr */
    while (current_row < num_block_rows) {
        row_ptr[current_row + 1] = nnz_blocks;
        current_row++;
    }

    /* ===== 阶段4: 填充块内数值 =====
     *
     * 先将所有块统一填充为随机复数(常规矩阵),不区分对角/非对角。
     * 块内 r×r 元素按行主序存储: block[bi*r + bk] = A[I*r+bi][J*r+bk]
     */
    int r = block_size;
    for (uint64_t i = 0; i < nnz_blocks; i++) {
        complex_float_t *block = &values[i * r * r];
        for (int bi = 0; bi < r; bi++) {
            for (int bk = 0; bk < r; bk++) {
                block[bi * r + bk].re = (float)rand() / RAND_MAX;
                block[bi * r + bk].im = (float)rand() / RAND_MAX;
            }
        }
    }

    /* ===== 阶段5: Hermitian 约束——对角块强制Hermitian对称 =====
     *
     * Hermitian 矩阵要求对角块(I==J)本身为 Hermitian:
     *   block[bi][bk] = conj(block[bk][bi]), 且 block[bi][bi] 为实数。
     * 遍历所有对角块,将块内下三角填充为上三角的共轭,对角线虚部置零。
     * 非对角块不受影响。
     */
    for (uint64_t I = 0; I < num_block_rows; I++) {
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            if ((uint64_t)col_idx[j] == I) {  /* 对角块 I==J */
                complex_float_t *block = &values[j * r * r];
                for (int bi = 0; bi < r; bi++) {
                    block[bi * r + bi].im = 0.0f;  /* 对角线虚部置零 */
                    for (int bk = bi + 1; bk < r; bk++) {
                        block[bk * r + bi].re = block[bi * r + bk].re;   /* 下三角 = 上三角共轭 */
                        block[bk * r + bi].im = -block[bi * r + bk].im;
                    }
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
