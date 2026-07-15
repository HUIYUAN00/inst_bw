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

static complex_float_t alpha = {1.0f, 0.0f};
static complex_float_t beta = {1.0f, 0.0f};
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
    uint64_t dim = nbr * r;

    complex_float_t *y_orig = y;
    complex_float_t *y_save = NULL;
    posix_memalign((void**)&y_save, 64, dim * sizeof(complex_float_t));
    memset(y_save, 0, dim * sizeof(complex_float_t));
    y = y_save;

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

    for (uint64_t i = 0; i < dim; i++) {
        complex_float_t ax = y[i];
        y_orig[i].re = alpha.re * ax.re - alpha.im * ax.im + beta.re * y_orig[i].re - beta.im * y_orig[i].im;
        y_orig[i].im = alpha.re * ax.im + alpha.im * ax.re + beta.re * y_orig[i].im + beta.im * y_orig[i].re;
    }
    free(y_save);
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
    uint64_t dim = nbr * r;
    uint64_t vl_floats = svcntw();  /* SVE向量寄存器可容纳的float32数 */
    svbool_t pgall = svptrue_b32();
    svfloat32_t zzero = svdup_f32(0.0f);
    uint64_t total = (uint64_t)r * 2;

    complex_float_t *y_orig = y;
    complex_float_t *y_save = NULL;
    posix_memalign((void**)&y_save, 64, dim * sizeof(complex_float_t));
    memset(y_save, 0, dim * sizeof(complex_float_t));
    y = y_save;

    for (uint64_t I = 0; I < nbr; I++) {

        /* ---- 对角块 (J==I): 块内 Hermitian,仅读上三角(bi<=bk) ----
         *
         * Hermitian性质: block[bk][bi] = conj(block[bi][bk])
         *   对角线(bi==bk): block[bi][bi] 为实数(虚部=0)
         *   下三角(bi>bk): 可由上三角共轭推导,无需存储
         *
         * 循环结构: r维循环(bi)为外层, for(bj)循环为内层。
         *   对角线元素(bk==bi)单独标量处理: y[bi] += block[bi][bi] * x[bi]
         *   循环从 bj=bi+1 开始,每次步进 svcntd() 个复数(64-bit)元素,
         *   svwhilelt_b32(bj*2, r*2) 自动处理尾部不足一个寄存器的情况。
         *
         *   (1) 内积(bk>bi): y[bi] += block[bi][bk] * x[bk]
         *       svcmla #0 + #90 完成复数乘法 a*x, 累加到 zacc
         *       循环结束后统一归约 zacc → y[bi]
         *
         *   (2) 外积(bk>bi): y[bk] += conj(block[bi][bk]) * x[bi]
         *       svcmla #0 + #270 计算 conj(a)*x:
         *         #0:   [re += ar*xre,    im += ar*xim]
         *         #270: [re += ai*xim,    im += -ai*xre]
         *         合计: re += ar*xre+ai*xim, im += ar*xim-ai*xre ✓
         *       原地读改写 zy, 仅写回 pg 激活 lane
         */
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            int32_t J = col_idx[j];
            if ((uint64_t)J != I) continue;
            complex_float_t *block = &val[j * r * r];
            complex_float_t *xi = &vec[I * r];
            complex_float_t *yi = &y[I * r];

            for (int bi = 0; bi < r; bi++) {
                float *br = (float *)&block[bi * r];
                float *xr = (float *)xi;
                float *yr = (float *)yi;

                /* 对角线元素(bk==bi): 标量内积, block[bi][bi]为实数(虚部=0) */
                complex_float_t a_diag = block[bi * r + bi];
                yi[bi].re += a_diag.re * xi[bi].re;
                yi[bi].im += a_diag.re * xi[bi].im;

                /* 外积广播向量: x[bi] 视为 1 个 double (2个float=1个复数), svdup_f64 广播
                 * reinterpret 为 f32 后 = [re,im,re,im,...], 直接用于 FCMLA */
                svfloat32_t zx_bcast = svreinterpret_f32_f64(svdup_f64(*(const double *)&xi[bi]));

                /* 内积累加器: 跨循环迭代累加,循环结束后统一归约 */
                svfloat32_t zacc = svdup_f32(0.0f);

                /* 循环: bj从bi+1开始,每次处理svcntd()个复数(64-bit)元素
                 * bj*2 为 float 偏移, svwhilelt_b32(bj*2, r*2) 生成32-bit谓词直接用于FCMLA */
                uint64_t vld = svcntd();
                for (int64_t bj = bi + 1; bj < r; bj += vld) {
                    svbool_t pg = svwhilelt_b32(bj * 2, (int64_t)r * 2);

                    svfloat32_t za = svld1_f32(pg, br + bj * 2);
                    svfloat32_t zx = svld1_f32(pg, xr + bj * 2);

                    /* (1) 内积(bk>bi): zacc += a * x[bk] */
                    zacc = svcmla_f32_m(pg, zacc, za, zx, 0);
                    zacc = svcmla_f32_m(pg, zacc, za, zx, 90);

                    /* (2) 外积(bk>bi): y[bk] += conj(a) * x[bi], 原地读改写 zy */
                    svfloat32_t zy = svld1_f32(pg, yr + bj * 2);
                    zy = svcmla_f32_m(pg, zy, za, zx_bcast, 0);
                    zy = svcmla_f32_m(pg, zy, za, zx_bcast, 270);
                    svst1_f32(pg, yr + bj * 2, zy);
                }

                /* 内积累加器归约: zacc=[re,im,re,im,...] → sum_re, sum_im → y[bi] */
                svfloat32_t zre = svuzp1_f32(zacc, zzero);
                svfloat32_t zim = svuzp2_f32(zacc, zzero);
                yi[bi].re += svaddv_f32(pgall, zre);
                yi[bi].im += svaddv_f32(pgall, zim);
            }
        }

        /* ---- 上三角块 (J>I): 内积 y[I*r+i] += block[i][k] * x[J*r+k] ----
         * 4块循环展开: 一次迭代处理4个不同block,4路FCMLA累加后合并归约
         * col_idx行内有序,跳过对角块后剩余均为上三角,4块一组处理,尾部单独兜底
         * svcmla #0 + #90 完成复数乘法 a*x, 4路累加器 svadd 合并后 svuzp1/svuzp2 + svaddv 归约 */
        {
            uint64_t j = row_ptr[I];
            uint64_t j_end = row_ptr[I + 1];
            while (j < j_end && (uint64_t)col_idx[j] <= I) j++;

            for (; j + 3 < j_end; j += 4) {
                int32_t J0 = col_idx[j], J1 = col_idx[j + 1], J2 = col_idx[j + 2], J3 = col_idx[j + 3];
                complex_float_t *block0 = &val[j * r * r];
                complex_float_t *block1 = &val[(j + 1) * r * r];
                complex_float_t *block2 = &val[(j + 2) * r * r];
                complex_float_t *block3 = &val[(j + 3) * r * r];
                float *xr0 = (float *)&vec[J0 * r];
                float *xr1 = (float *)&vec[J1 * r];
                float *xr2 = (float *)&vec[J2 * r];
                float *xr3 = (float *)&vec[J3 * r];

                uint64_t off = 0;
                while (off < total) {
                    uint64_t cnt = (total - off < vl_floats) ? (total - off) : vl_floats;
                    svbool_t pg = svcmplt_u32(pgall, svindex_u32(0, 1), svdup_u32(cnt));
                    svfloat32_t zx0 = svld1_f32(pg, xr0 + off);
                    svfloat32_t zx1 = svld1_f32(pg, xr1 + off);
                    svfloat32_t zx2 = svld1_f32(pg, xr2 + off);
                    svfloat32_t zx3 = svld1_f32(pg, xr3 + off);

                    for (int i = 0; i < r; i++) {
                        svfloat32_t za0 = svld1_f32(pg, (float *)&block0[i * r] + off);
                        svfloat32_t za1 = svld1_f32(pg, (float *)&block1[i * r] + off);
                        svfloat32_t za2 = svld1_f32(pg, (float *)&block2[i * r] + off);
                        svfloat32_t za3 = svld1_f32(pg, (float *)&block3[i * r] + off);

                        svfloat32_t zacc0 = svdup_f32(0.0f);
                        svfloat32_t zacc1 = svdup_f32(0.0f);
                        svfloat32_t zacc2 = svdup_f32(0.0f);
                        svfloat32_t zacc3 = svdup_f32(0.0f);
                        zacc0 = svcmla_f32_m(pg, zacc0, za0, zx0, 0);
                        zacc0 = svcmla_f32_m(pg, zacc0, za0, zx0, 90);
                        zacc1 = svcmla_f32_m(pg, zacc1, za1, zx1, 0);
                        zacc1 = svcmla_f32_m(pg, zacc1, za1, zx1, 90);
                        zacc2 = svcmla_f32_m(pg, zacc2, za2, zx2, 0);
                        zacc2 = svcmla_f32_m(pg, zacc2, za2, zx2, 90);
                        zacc3 = svcmla_f32_m(pg, zacc3, za3, zx3, 0);
                        zacc3 = svcmla_f32_m(pg, zacc3, za3, zx3, 90);

                        /* 4路累加器合并后统一归约 */
                        zacc0 = svadd_f32_x(pgall, zacc0, zacc1);
                        zacc2 = svadd_f32_x(pgall, zacc2, zacc3);
                        zacc0 = svadd_f32_x(pgall, zacc0, zacc2);
                        svfloat32_t zre = svuzp1_f32(zacc0, zzero);
                        svfloat32_t zim = svuzp2_f32(zacc0, zzero);
                        y[I * r + i].re += svaddv_f32(pgall, zre);
                        y[I * r + i].im += svaddv_f32(pgall, zim);
                    }
                    off += cnt;
                }
            }

            /* 尾部剩余块(不足4个) */
            for (; j < j_end; j++) {
                int32_t J = col_idx[j];
                complex_float_t *block = &val[j * r * r];
                float *xr = (float *)&vec[J * r];

                uint64_t off = 0;
                while (off < total) {
                    uint64_t cnt = (total - off < vl_floats) ? (total - off) : vl_floats;
                    svbool_t pg = svcmplt_u32(pgall, svindex_u32(0, 1), svdup_u32(cnt));
                    svfloat32_t zx = svld1_f32(pg, xr + off);

                    for (int i = 0; i < r; i++) {
                        float *br = (float *)&block[i * r];
                        svfloat32_t za = svld1_f32(pg, br + off);
                        svfloat32_t zacc = svdup_f32(0.0f);
                        zacc = svcmla_f32_m(pg, zacc, za, zx, 0);
                        zacc = svcmla_f32_m(pg, zacc, za, zx, 90);
                        svfloat32_t zre = svuzp1_f32(zacc, zzero);
                        svfloat32_t zim = svuzp2_f32(zacc, zzero);
                        y[I * r + i].re += svaddv_f32(pgall, zre);
                        y[I * r + i].im += svaddv_f32(pgall, zim);
                    }
                    off += cnt;
                }
            }
        }

        /* ---- 下三角贡献 (J>I): 共轭转置 y[J*r+k] += conj(block[i][k]) * x[I*r+i] ----
         * 向量化循环为外层(遍历k方向的向量chunk), r维循环为内层(遍历块各行i)
         * zy(y向量chunk) 在外层加载一次,内层 r 行共享累加,最后统一写回
         * 每行贡献 conj(block[i][k]) * x[i]
         *
         * conj(a)*x 实现 (FCMLA #0 + #270):
         *   x[i] 视为 64-bit (re,im 打包), svdup_u64 广播为 [xre,xim,xre,xim,...]
         *   FCMLA #0:   [re += ar*xre,    im += ar*xim]
         *   FCMLA #270: [re += ai*xim,    im += -ai*xre]
         *   合计: re += ar*xre+ai*xim, im += ar*xim-ai*xre ✓ = conj(a)*x */
        for (uint64_t j = row_ptr[I]; j < row_ptr[I + 1]; j++) {
            int32_t J = col_idx[j];
            if ((uint64_t)J <= I) continue;
            complex_float_t *block = &val[j * r * r];
            complex_float_t *xi = &vec[I * r];
            float *yr = (float *)&y[J * r];

            uint64_t off = 0;
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

    for (uint64_t i = 0; i < dim; i++) {
        complex_float_t ax = y[i];
        y_orig[i].re = alpha.re * ax.re - alpha.im * ax.im + beta.re * y_orig[i].re - beta.im * y_orig[i].im;
        y_orig[i].im = alpha.re * ax.im + alpha.im * ax.re + beta.re * y_orig[i].im + beta.im * y_orig[i].re;
    }
    free(y_save);
}

static test_item_t test_registry[] = {
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
    printf("  --alpha <re,im>         Complex alpha (default: 1.0,0.0)\n");
    printf("  --beta <re,im>          Complex beta (default: 1.0,0.0)\n");
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
        if (strcmp(argv[i], "--alpha") == 0) {
            if (i + 1 < argc) sscanf(argv[++i], "%f,%f", &alpha.re, &alpha.im);
            continue;
        }
        if (strcmp(argv[i], "--beta") == 0) {
            if (i + 1 < argc) sscanf(argv[++i], "%f,%f", &beta.re, &beta.im);
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
    MPI_Bcast(&alpha, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
        printf("Alpha: (%.6f, %.6f)\n", alpha.re, alpha.im);
        printf("Beta: (%.6f, %.6f)\n", beta.re, beta.im);
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

    /* 随机初始化 result/result_ref (相同值), 计算函数做 y += A*x 不再内部置零 */
    for (uint64_t i = 0; i < matrix_dim; i++) {
        result[i].re = (float)rand() / RAND_MAX;
        result[i].im = (float)rand() / RAND_MAX;
        result_ref[i] = result[i];
    }
    complex_float_t *result_init = (complex_float_t *)malloc(matrix_dim * sizeof(complex_float_t));
    memcpy(result_init, result, matrix_dim * sizeof(complex_float_t));

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

    /* ===== 正确性校验: 每个测试函数单独运行一次,与参考结果对比 ===== */
    if (rank == 0) {
        printf("Correctness Verification:\n");
        printf("================================================================================\n");
    }

    for (int i = 0; i < test_count; i++) {
        if (!run_all && !should_run_test(i, num_specs, specs)) continue;

        test_item_t *test = &test_registry[i];
        memcpy(result, result_init, matrix_dim * sizeof(complex_float_t));
        test->func(result, values, vector, nnz_blocks, 1.0);
        int verify_result = verify_hemv(result, result_ref);

        if (rank == 0) {
            printf("  %-38s ", test->name);
            if (verify_result > 0) {
                printf("FAIL(%d)\n", verify_result);
            } else {
                printf("PASS\n");
            }
        }
    }

    /* ===== 性能测试 ===== */
    if (rank == 0) {
        printf("\n");
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

        memcpy(result, result_init, matrix_dim * sizeof(complex_float_t));

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
    free(result_init);

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
