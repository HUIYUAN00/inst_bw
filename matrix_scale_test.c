#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static int matrix_dim = 1024;
static double alpha = 2.0;
static int warmup_iter = 5;
static int test_iter = 10;
static unsigned int random_seed = 42;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

static void matrix_scale(double *A, double *B, int n, double alpha) {
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (; j < n - 3; j += 4) {
            B[i * n + j] = A[i * n + j] * alpha;
            B[i * n + j + 1] = A[i * n + j + 1] * alpha;
            B[i * n + j + 2] = A[i * n + j + 2] * alpha;
            B[i * n + j + 3] = A[i * n + j + 3] * alpha;
        }
        for (; j < n; j++) {
            B[i * n + j] = A[i * n + j] * alpha;
        }
    }
}

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -n, --dim <N>           Matrix dimension N (default: 1024)\n");
    printf("  -a, --alpha <value>     Scaling factor alpha (default: 2.0)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("  -r, --random-seed <N>   Random seed for data initialization (default: 42)\n");
}

static double run_test(double *A, double *B, int n, double alpha
#ifdef USE_MPI
    , MPI_Comm comm
#endif
) {
    struct timeval start, end;
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    for (int i = 0; i < warmup_iter; i++) {
        matrix_scale(A, B, n, alpha);
    }
    
#ifdef USE_MPI
    MPI_Barrier(comm);
#endif
    
    gettimeofday(&start, NULL);
    for (int i = 0; i < test_iter; i++) {
        matrix_scale(A, B, n, alpha);
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
        if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--dim") == 0) {
            if (i + 1 < argc) {
                matrix_dim = atoi(argv[++i]);
                if (matrix_dim < 1) matrix_dim = 1024;
            }
            continue;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--alpha") == 0) {
            if (i + 1 < argc) {
                alpha = atof(argv[++i]);
            }
            continue;
        }
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 < argc) {
                warmup_iter = atoi(argv[++i]);
                if (warmup_iter < 0) warmup_iter = 5;
            }
            continue;
        }
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            if (i + 1 < argc) {
                test_iter = atoi(argv[++i]);
                if (test_iter < 1) test_iter = 10;
            }
            continue;
        }
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random-seed") == 0) {
            if (i + 1 < argc) {
                random_seed = (unsigned int)atoi(argv[++i]);
            }
            continue;
        }
    }
    
#ifdef USE_MPI
    MPI_Bcast(&matrix_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&warmup_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    
    uint64_t matrix_bytes = (uint64_t)matrix_dim * matrix_dim * sizeof(double);
    uint64_t bytes_per_iter = matrix_bytes * 2;
    
    double *A = NULL, *B = NULL;
    
    A = (double *)malloc(matrix_bytes);
    B = (double *)malloc(matrix_bytes);
    if (!A || !B) {
#ifdef USE_MPI
        fprintf(stderr, "[Rank %d] Failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        fprintf(stderr, "Failed to allocate memory\n");
#endif
        return 1;
    }
    
    srand(random_seed);
    
    for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
            A[i * matrix_dim + j] = (double)rand() / RAND_MAX;
            B[i * matrix_dim + j] = (double)rand() / RAND_MAX;
        }
    }
    
    if (rank == 0) {
        printf("================================================================================\n");
#ifdef USE_MPI
        printf("Matrix Scale Benchmark (MPI - %d processes)\n", nprocs);
#else
        printf("Matrix Scale Benchmark\n");
#endif
        printf("================================================================================\n");
        printf("Matrix Dimension: %d x %d\n", matrix_dim, matrix_dim);
        printf("Matrix Size: %.2f MB per matrix\n", (double)matrix_bytes / (1024 * 1024));
        printf("Alpha: %.2f\n", alpha);
        printf("Warmup Iterations: %d\n", warmup_iter);
        printf("Test Iterations: %d\n", test_iter);
        printf("Random Seed: %u\n", random_seed);
        printf("Data Per Iteration: %.2f MB (Read+Write)\n", (double)bytes_per_iter / (1024 * 1024));
        printf("\n");
    }
    
#ifdef USE_MPI
    double time_sec = run_test(A, B, matrix_dim, alpha, MPI_COMM_WORLD);
#else
    double time_sec = run_test(A, B, matrix_dim, alpha);
#endif
    
    double bandwidth = get_bandwidth(bytes_per_iter, time_sec);
    
#ifdef USE_MPI
    double total_bw = 0.0;
    MPI_Reduce(&bandwidth, &total_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    
    if (rank == 0) {
#ifdef USE_MPI
        printf("%-20s %10s %10s %10s %10s\n", "Test", "GB/s", "Time(ms)", "Data(MB)", "Total(GB/s)");
        printf("%-20s %10.2f %10.3f %10.0f %10.2f\n", 
               "B = A * alpha", bandwidth, time_sec * 1000,
               (double)bytes_per_iter / (1024 * 1024), total_bw);
#else
        printf("%-20s %10s %10s %10s\n", "Test", "GB/s", "Time(ms)", "Data(MB)");
        printf("%-20s %10.2f %10.3f %10.0f\n", 
               "B = A * alpha", bandwidth, time_sec * 1000,
               (double)bytes_per_iter / (1024 * 1024));
#endif
        printf("================================================================================\n");
    }
    
    free(A);
    free(B);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}