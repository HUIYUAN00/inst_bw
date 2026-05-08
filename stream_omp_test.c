#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static int warmup_iter = 5;
static int test_iter = 10;
static uint64_t buffer_size = 128 * 1024 * 1024;
static int num_threads = 1;

#define MEM_ALIGNMENT 64

typedef struct {
    const char *name;
    const char *category;
    void (*func)(double *a, double *b, double *c, uint64_t size, double scalar, int threads);
    uint64_t bytes_per_iter;
} test_item_t;

static inline double get_bandwidth(uint64_t bytes, double time_sec) {
    return bytes / time_sec / 1e9;
}

static void *alloc_buffer(uint64_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, MEM_ALIGNMENT, size) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory (size=%lu, alignment=%d)\n", size, MEM_ALIGNMENT);
        return NULL;
    }
    return ptr;
}

static void free_buffer(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

#pragma GCC push_options
#pragma GCC optimize ("O3")

static void stream_copy(double *a, double *b, double *c, uint64_t size, double scalar, int threads) {
    uint64_t count = size / sizeof(double);
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        uint64_t start = tid * count / nth;
        uint64_t end = (tid + 1) * count / nth;
        for (uint64_t i = start; i < end; i++) {
            a[i] = b[i];
        }
    }
}

static void stream_scale(double *a, double *b, double *c, uint64_t size, double scalar, int threads) {
    uint64_t count = size / sizeof(double);
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        uint64_t start = tid * count / nth;
        uint64_t end = (tid + 1) * count / nth;
        for (uint64_t i = start; i < end; i++) {
            a[i] = b[i] * scalar;
        }
    }
}

static void stream_add(double *a, double *b, double *c, uint64_t size, double scalar, int threads) {
    uint64_t count = size / sizeof(double);
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        uint64_t start = tid * count / nth;
        uint64_t end = (tid + 1) * count / nth;
        for (uint64_t i = start; i < end; i++) {
            a[i] = b[i] + c[i];
        }
    }
}

static void stream_triad(double *a, double *b, double *c, uint64_t size, double scalar, int threads) {
    uint64_t count = size / sizeof(double);
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        uint64_t start = tid * count / nth;
        uint64_t end = (tid + 1) * count / nth;
        for (uint64_t i = start; i < end; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
}

#pragma GCC pop_options

static test_item_t test_registry[] = {
    {"STREAM Copy",  "STREAM",  stream_copy,  0},
    {"STREAM Scale", "STREAM",  stream_scale, 0},
    {"STREAM Add",   "STREAM",  stream_add,   0},
    {"STREAM Triad", "STREAM",  stream_triad, 0},
};

static const int test_count = sizeof(test_registry) / sizeof(test_registry[0]);

static void print_usage(const char *prog_name) {
    printf("Usage: %s [options] [test_spec...]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -l, --list              List all available tests\n");
    printf("  -b, --buffer-size <MB>  Buffer size in MB (default: 128)\n");
    printf("  -n, --threads <N>       Number of threads (default: 1)\n");
    printf("  -w, --warmup <N>        Warmup iterations (default: 5)\n");
    printf("  -t, --test <N>          Test iterations (default: 10)\n");
    printf("\nTest Specification:\n");
    printf("  <index>                 Run test by index (0-based)\n");
    printf("  <name>                  Run test by name (partial match)\n");
    printf("\nExamples:\n");
    printf("  %s                               Run all tests (1 thread)\n", prog_name);
    printf("  %s -n 4                          Run with 4 threads\n", prog_name);
    printf("  %s -b 256 -n 8                   256MB buffer, 8 threads\n", prog_name);
    printf("  %s -n 4 Triad                    Run Triad with 4 threads\n", prog_name);
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

static double run_test(test_item_t *test, double *a, double *b, double *c) {
    double start_time, end_time;
    double scalar = 2.0;
    
    for (int i = 0; i < warmup_iter; i++) {
        test->func(a, b, c, buffer_size, scalar, num_threads);
    }
    
    start_time = omp_get_wtime();
    for (int i = 0; i < test_iter; i++) {
        test->func(a, b, c, buffer_size, scalar, num_threads);
    }
    end_time = omp_get_wtime();
    
    double time_sec = (end_time - start_time) / test_iter;
    return time_sec;
}

int main(int argc, char *argv[]) {
    int run_all = 1;
    int num_specs = 0;
    char **specs = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            print_tests();
            return 0;
        }
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--buffer-size") == 0) {
            if (i + 1 < argc) {
                buffer_size = (uint64_t)atoi(argv[++i]) * 1024 * 1024;
            }
            continue;
        }
        if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                num_threads = atoi(argv[++i]);
                if (num_threads < 1) num_threads = 1;
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
    
    buffer_size = (buffer_size / 1024) * 1024;
    if (buffer_size < 1024) buffer_size = 1024;
    
    printf("============================================================\n");
    printf("STREAM Multi-threaded Bandwidth Benchmark (OpenMP)\n");
    printf("============================================================\n");
    printf("Buffer Size: %lu MB per array\n", buffer_size / (1024 * 1024));
    printf("Number of Threads: %d\n", num_threads);
    printf("Warmup Iterations: %d\n", warmup_iter);
    printf("Test Iterations: %d\n", test_iter);
    printf("Registered Tests: %d\n\n", test_count);
    
    double *a = NULL, *b = NULL, *c = NULL;
    
    a = (double *)alloc_buffer(buffer_size);
    b = (double *)alloc_buffer(buffer_size);
    c = (double *)alloc_buffer(buffer_size);
    
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Failed to allocate buffers\n");
        if (a) free_buffer(a);
        if (b) free_buffer(b);
        if (c) free_buffer(c);
        return 1;
    }
    
    uint64_t elem_count = buffer_size / sizeof(double);
    for (uint64_t i = 0; i < elem_count; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }
    
    printf("%-22s %10s %10s %10s %10s\n", 
           "Test", "Category", "GB/s", "Time(ms)", "Data(MB)");
    printf("============================================================\n");
    
    for (int i = 0; i < test_count; i++) {
        if (!run_all && !should_run_test(i, num_specs, specs)) continue;
        
        test_item_t *test = &test_registry[i];
        
        uint64_t bytes_per_iter = buffer_size;
        if (strstr(test->name, "Copy") != NULL || strstr(test->name, "Scale") != NULL) {
            bytes_per_iter = buffer_size * 2;
        } else if (strstr(test->name, "Add") != NULL || strstr(test->name, "Triad") != NULL) {
            bytes_per_iter = buffer_size * 3;
        }
        
        double time_sec = run_test(test, a, b, c);
        double bandwidth = get_bandwidth(bytes_per_iter, time_sec);
        
        printf("%-22s %10s %10.2f %10.3f %10.0f\n",
               test->name, test->category, bandwidth, time_sec * 1000,
               (double)bytes_per_iter / (1024 * 1024));
    }
    
    printf("============================================================\n");
    
    free_buffer(a);
    free_buffer(b);
    free_buffer(c);
    
    return 0;
}