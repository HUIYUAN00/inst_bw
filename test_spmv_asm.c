#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct {
    double re;
    double im;
} complex_double_t;

typedef struct {
    uint32_t row;
    uint32_t col;
    complex_double_t value;
} coo_entry_t;

extern void spmv_standard(void *result_ptr, void *values_ptr, void *vector_ptr,
                          uint64_t *row_ptr, int32_t *col_idx, int matrix_dim);

static void hermitian_spmv_scalar(void *result_ptr, void *values_ptr, void *vector_ptr,
                                   uint64_t *row_ptr, int32_t *col_idx, int matrix_dim) {
    complex_double_t *val = (complex_double_t *)values_ptr;
    complex_double_t *vec = (complex_double_t *)vector_ptr;
    complex_double_t *y = (complex_double_t *)result_ptr;

    for (int i = 0; i < matrix_dim; i++) {
        y[i].re = 0.0;
        y[i].im = 0.0;
    }

    for (int i = 0; i < matrix_dim; i++) {
        for (uint64_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            int32_t col = col_idx[j];
            if (col < i) continue;

            complex_double_t a = val[j];
            complex_double_t xj = vec[col];

            y[i].re += a.re * xj.re - a.im * xj.im;
            y[i].im += a.re * xj.im + a.im * xj.re;

            if (col != i) {
                complex_double_t xi = vec[i];
                y[col].re += a.re * xi.re + a.im * xi.im;
                y[col].im += a.re * xi.im - a.im * xi.re;
            }
        }
    }
}

static int compare_coo(const void *a, const void *b) {
    const coo_entry_t *ca = (const coo_entry_t *)a;
    const coo_entry_t *cb = (const coo_entry_t *)b;
    if (ca->row != cb->row) return (ca->row < cb->row) ? -1 : 1;
    return (ca->col < cb->col) ? -1 : (ca->col > cb->col) ? 1 : 0;
}

static void generate_hermitian_matrix(int matrix_dim, double sparsity, unsigned int seed,
                                       uint64_t *row_ptr, int32_t *col_idx,
                                       complex_double_t *values, uint64_t *out_nnz) {
    uint64_t total_elements = (uint64_t)matrix_dim * matrix_dim;
    uint64_t nnz = (uint64_t)(total_elements * sparsity);
    if (nnz < (uint64_t)matrix_dim) nnz = matrix_dim;

    coo_entry_t *coo = (coo_entry_t *)malloc(nnz * sizeof(coo_entry_t));
    uint64_t *coverage = (uint64_t *)calloc((total_elements / 64) + 2, sizeof(uint64_t));
    uint64_t unique_count = 0;
    int attempts = 0;
    int max_attempts = nnz * 20;

    srand(seed);

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
    qsort(coo, unique_count, sizeof(coo_entry_t), compare_coo);

    row_ptr[0] = 0;
    uint64_t current_row = 0;

    for (uint64_t i = 0; i < unique_count; i++) {
        col_idx[i] = (int32_t)coo[i].col;
        values[i] = coo[i].value;

        while (current_row < coo[i].row) {
            row_ptr[current_row + 1] = i;
            current_row++;
        }
    }

    while (current_row < (uint64_t)matrix_dim) {
        row_ptr[current_row + 1] = unique_count;
        current_row++;
    }

    *out_nnz = unique_count;
    free(coo);
}

int main(int argc, char *argv[]) {
    int matrix_dim = 16;
    double sparsity = 0.1;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) matrix_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = atof(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) seed = (unsigned int)atoi(argv[++i]);
    }

    uint64_t total_elements = (uint64_t)matrix_dim * matrix_dim;
    uint64_t nnz = (uint64_t)(total_elements * sparsity);
    if (nnz < (uint64_t)matrix_dim) nnz = matrix_dim;

    uint64_t *row_ptr = (uint64_t *)malloc((matrix_dim + 1) * sizeof(uint64_t));
    int32_t *col_idx = (int32_t *)malloc(nnz * sizeof(int32_t));
    complex_double_t *values = (complex_double_t *)malloc(nnz * sizeof(complex_double_t));
    complex_double_t *vector = (complex_double_t *)malloc(matrix_dim * sizeof(complex_double_t));
    complex_double_t *result = (complex_double_t *)malloc(matrix_dim * sizeof(complex_double_t));
    complex_double_t *result_ref = (complex_double_t *)malloc(matrix_dim * sizeof(complex_double_t));

    if (!row_ptr || !col_idx || !values || !vector || !result || !result_ref) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    srand(seed);
    for (int i = 0; i < matrix_dim; i++) {
        vector[i].re = (double)rand() / RAND_MAX;
        vector[i].im = (double)rand() / RAND_MAX;
    }

    uint64_t actual_nnz;
    generate_hermitian_matrix(matrix_dim, sparsity, seed, row_ptr, col_idx, values, &actual_nnz);

    printf("Matrix: %dx%d, NNZ: %lu, Avg NNZ/Row: %.2f\n",
           matrix_dim, matrix_dim, actual_nnz, (double)actual_nnz / matrix_dim);

    hermitian_spmv_scalar(result_ref, values, vector, row_ptr, col_idx, matrix_dim);
    spmv_standard(result, values, vector, row_ptr, col_idx, matrix_dim);

    int errors = 0;
    for (int i = 0; i < matrix_dim; i++) {
        double re_diff = fabs(result[i].re - result_ref[i].re);
        double im_diff = fabs(result[i].im - result_ref[i].im);
        if (re_diff > 1e-9 || im_diff > 1e-9) {
            if (errors < 5) {
                fprintf(stderr, "MISMATCH y[%d]: expected (%.6f, %.6f), got (%.6f, %.6f)\n",
                        i, result_ref[i].re, result_ref[i].im, result[i].re, result[i].im);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL: %d mismatches\n", errors);
    }

    free(row_ptr);
    free(col_idx);
    free(values);
    free(vector);
    free(result);
    free(result_ref);

    return errors > 0 ? 1 : 0;
}
