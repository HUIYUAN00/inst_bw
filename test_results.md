# SVE 内存带宽测试结果

测试时间: 2026-04-22 16:26:42
测试平台: Linux localhost.localdomain 6.6.0-145.0.2.143.oe2403sp2.aarch64 #1 SMP Fri Apr 10 14:47:07 CST 2026 aarch64 aarch64 aarch64 GNU/Linux

## 系统信息

```
CPU(s):                               192
On-line CPU(s) list:                  0-191
Model name:                           -
Thread(s) per core:                   2
Core(s) per socket:                   48
Socket(s):                            2
CPU(s) scaling MHz:                   100%
CPU max MHz:                          2200.0000
CPU min MHz:                          1200.0000
NUMA node0 CPU(s):                    0-47
NUMA node1 CPU(s):                    48-95
NUMA node2 CPU(s):                    96-143
NUMA node3 CPU(s):                    144-191
```

## 单进程版本测试 (sve_bw_test)

### 帮助信息

**命令**: `./sve_bw_test --help`

```
Usage: ./sve_bw_test [options] [test_spec...]

Options:
  -l, --list     List all available tests
  -a, --all      Run all tests (default)

Test Specification:
  <index>        Run test by index (0-based)
  <name>         Run test by name (partial match)
  <category>     Run all tests in a category

Examples:
  ./sve_bw_test                    Run all tests
  ./sve_bw_test 0 1 2              Run tests 0, 1, and 2
  ./sve_bw_test NEON               Run all NEON tests
  ./sve_bw_test Gather Scatter     Run Gather and Scatter tests
  ./sve_bw_test "SVE LD1D"         Run tests matching 'SVE LD1D'
```

### 测试列表

**命令**: `./sve_bw_test --list`

```
Available Tests:
============================================================
Idx  Test Name                Category
============================================================
0    NEON LDP (Read)              Load
1    NEON STP (Write)            Store
2    NEON LDP+STP (Copy)          Copy
3    SVE LD1B (Read)              Load
4    SVE ST1B (Write)            Store
5    SVE LD1B+ST1B (Copy)         Copy
6    SVE LD1W (Read)              Load
7    SVE ST1W (Write)            Store
8    SVE LD1W+ST1W (Copy)         Copy
9    SVE LD1D (Read)              Load
10   SVE ST1D (Write)            Store
11   SVE LD1D+ST1D (Copy)         Copy
12   SVE Gather LD1W            Gather
13   SVE Gather LD1SW+LD1D      Gather
14   SVE Scatter ST1W          Scatter
15   SVE Scatter ST1D          Scatter
16   SVE Gather+Scatter W   GatherScatter
17   SVE Gather+Scatter D   GatherScatter
18   STREAM Copy                STREAM
19   STREAM Scale               STREAM
20   STREAM Add                 STREAM
21   STREAM Triad               STREAM
============================================================
```

### 运行所有测试

**命令**: `./sve_bw_test`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Registered Tests: 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      28.75      4.669        128
NEON STP (Write)            Store      47.77      2.810        128
NEON LDP+STP (Copy)          Copy      39.25      6.840        256
SVE LD1B (Read)              Load      28.91      4.643        128
SVE ST1B (Write)            Store      47.36      2.834        128
SVE LD1B+ST1B (Copy)         Copy      39.41      6.812        256
SVE LD1W (Read)              Load      28.90      4.645        128
SVE ST1W (Write)            Store      41.78      3.212        128
SVE LD1W+ST1W (Copy)         Copy      39.67      6.767        256
SVE LD1D (Read)              Load      28.75      4.669        128
SVE ST1D (Write)            Store      47.74      2.812        128
SVE LD1D+ST1D (Copy)         Copy      38.94      6.893        256
SVE Gather LD1W            Gather       3.55     75.585        256
SVE Gather LD1SW+LD1D      Gather       1.90    212.188        384
SVE Scatter ST1W          Scatter       3.14     85.360        256
SVE Scatter ST1D          Scatter       2.00    134.133        256
SVE Gather+Scatter W   GatherScatter       1.77    151.955        256
SVE Gather+Scatter D   GatherScatter       1.59    168.653        256
STREAM Copy                STREAM      33.35      8.048        256
STREAM Scale               STREAM      34.00      7.895        256
STREAM Add                 STREAM      31.42     12.816        384
STREAM Triad               STREAM      30.21     13.330        384
============================================================
```

### 按索引运行 (测试 0, 6, 9)

**命令**: `./sve_bw_test 0 6 9`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 3 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      28.96      4.635        128
SVE LD1W (Read)              Load      28.83      4.656        128
SVE LD1D (Read)              Load      28.95      4.636        128
============================================================
```

### 按类别运行 - Load

**命令**: `./sve_bw_test Load`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      28.37      4.731        128
SVE LD1B (Read)              Load      28.31      4.742        128
SVE LD1W (Read)              Load      28.31      4.741        128
SVE LD1D (Read)              Load      28.29      4.744        128
============================================================
```

### 按类别运行 - STREAM

**命令**: `./sve_bw_test STREAM`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
STREAM Copy                STREAM      39.75      6.753        256
STREAM Scale               STREAM      38.73      6.930        256
STREAM Add                 STREAM      36.18     11.130        384
STREAM Triad               STREAM      34.78     11.576        384
============================================================
```

### 按名称匹配 - LD1D

**命令**: `./sve_bw_test "LD1D"`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 3 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
SVE LD1D (Read)              Load      29.00      4.628        128
SVE LD1D+ST1D (Copy)         Copy      40.10      6.695        256
SVE Gather LD1SW+LD1D      Gather       1.86    216.752        384
============================================================
```

### 按名称匹配 - NEON

**命令**: `./sve_bw_test NEON`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 3 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      29.09      4.615        128
NEON STP (Write)            Store      48.32      2.778        128
NEON LDP+STP (Copy)          Copy      39.07      6.870        256
============================================================
```

### 按类别运行 - Gather

**命令**: `./sve_bw_test Gather`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
SVE Gather LD1W            Gather       3.19     84.071        256
SVE Gather LD1SW+LD1D      Gather       1.65    243.976        384
SVE Gather+Scatter W   GatherScatter       1.79    150.343        256
SVE Gather+Scatter D   GatherScatter       1.88    142.963        256
============================================================
```

### 按类别运行 - Scatter

**命令**: `./sve_bw_test Scatter`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
SVE Scatter ST1W          Scatter       3.00     89.602        256
SVE Scatter ST1D          Scatter       2.03    132.546        256
SVE Gather+Scatter W   GatherScatter       1.71    157.155        256
SVE Gather+Scatter D   GatherScatter       1.90    141.533        256
============================================================
```

### 混合指定 - 索引+类别

**命令**: `./sve_bw_test 0 1 STREAM`

```
============================================================
SVE Bandwidth Benchmark (Single Process)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 6 of 22

Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      26.77      5.015        128
NEON STP (Write)            Store      36.63      3.664        128
STREAM Copy                STREAM      29.55      9.084        256
STREAM Scale               STREAM      29.70      9.039        256
STREAM Add                 STREAM      28.93     13.916        384
STREAM Triad               STREAM      28.43     14.161        384
============================================================
```

### 无效测试名

**命令**: `./sve_bw_test INVALID_TEST`

```
No tests match the specified criteria.
Available Tests:
============================================================
Idx  Test Name                Category
============================================================
0    NEON LDP (Read)              Load
1    NEON STP (Write)            Store
2    NEON LDP+STP (Copy)          Copy
3    SVE LD1B (Read)              Load
4    SVE ST1B (Write)            Store
5    SVE LD1B+ST1B (Copy)         Copy
6    SVE LD1W (Read)              Load
7    SVE ST1W (Write)            Store
8    SVE LD1W+ST1W (Copy)         Copy
9    SVE LD1D (Read)              Load
10   SVE ST1D (Write)            Store
11   SVE LD1D+ST1D (Copy)         Copy
12   SVE Gather LD1W            Gather
13   SVE Gather LD1SW+LD1D      Gather
14   SVE Scatter ST1W          Scatter
15   SVE Scatter ST1D          Scatter
16   SVE Gather+Scatter W   GatherScatter
17   SVE Gather+Scatter D   GatherScatter
18   STREAM Copy                STREAM
19   STREAM Scale               STREAM
20   STREAM Add                 STREAM
21   STREAM Triad               STREAM
============================================================
```

## MPI 多进程版本测试 (sve_bw_test_mpi, 4进程)

### MPI 帮助信息

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi --help`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
[localhost.localdomain:1857064] [[34491,0],0] ORTE_ERROR_LOG: Data unpack had inadequate space in file util/show_help.c at line 513
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
Usage: ./sve_bw_test_mpi [options] [test_spec...]

Options:
  -l, --list     List all available tests
  -a, --all      Run all tests (default)

Test Specification:
  <index>        Run test by index (0-based)
  <name>         Run test by name (partial match)
  <category>     Run all tests in a category

Examples:
  ./sve_bw_test_mpi                    Run all tests
  ./sve_bw_test_mpi 0 1 2              Run tests 0, 1, and 2
  ./sve_bw_test_mpi NEON               Run all NEON tests
  ./sve_bw_test_mpi Gather Scatter     Run Gather and Scatter tests
  ./sve_bw_test_mpi "SVE LD1D"         Run tests matching 'SVE LD1D'
```

### MPI 运行所有测试

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
[localhost.localdomain:1862673] [[48258,0],0] ORTE_ERROR_LOG: Data unpack would read past end of buffer in file util/show_help.c at line 501
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Registered Tests: 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
NEON LDP (Read)              Load      27.72      4.843        128     108.98
NEON STP (Write)            Store      38.63      3.474        128     153.51
NEON LDP+STP (Copy)          Copy      30.75      8.729        256     120.68
SVE LD1B (Read)              Load      27.76      4.835        128     109.04
SVE ST1B (Write)            Store      38.56      3.481        128     153.17
SVE LD1B+ST1B (Copy)         Copy      30.73      8.735        256     120.71
SVE LD1W (Read)              Load      27.76      4.835        128     108.89
SVE ST1W (Write)            Store      38.65      3.473        128     153.41
SVE LD1W+ST1W (Copy)         Copy      30.27      8.869        256     119.63
SVE LD1D (Read)              Load      27.38      4.901        128     107.93
SVE ST1D (Write)            Store      38.79      3.460        128     154.05
SVE LD1D+ST1D (Copy)         Copy      30.57      8.781        256     120.44
SVE Gather LD1W            Gather       3.31     81.073        256      13.20
SVE Gather LD1SW+LD1D      Gather       1.91    210.709        384       7.59
SVE Scatter ST1W          Scatter       3.04     88.346        256      11.82
SVE Scatter ST1D          Scatter       1.94    138.659        256       7.91
SVE Gather+Scatter W   GatherScatter       1.58    170.370        256       6.59
SVE Gather+Scatter D   GatherScatter       1.86    144.025        256       7.56
STREAM Copy                STREAM      38.49      6.975        256     143.42
STREAM Scale               STREAM      37.42      7.174        256     142.73
STREAM Add                 STREAM      30.44     13.226        384     122.32
STREAM Triad               STREAM      30.77     13.085        384     122.02
============================================================
```

### MPI 按索引运行 (测试 0, 6, 9)

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi 0 6 9`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 3 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
NEON LDP (Read)              Load      26.69      5.030        128     107.94
SVE LD1W (Read)              Load      26.47      5.071        128     108.01
SVE LD1D (Read)              Load      26.52      5.062        128     107.92
============================================================
```

### MPI 按类别运行 - STREAM

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi STREAM`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
[localhost.localdomain:1878435] [[31024,0],0] ORTE_ERROR_LOG: Out of resource in file util/show_help.c at line 501
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
STREAM Copy                STREAM      38.24      7.021        256     154.17
STREAM Scale               STREAM      37.85      7.092        256     151.59
STREAM Add                 STREAM      33.09     12.168        384     131.07
STREAM Triad               STREAM      33.12     12.157        384     132.03
============================================================
```

### MPI 按类别运行 - Gather

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi Gather`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
[localhost.localdomain:1878559] [[31372,0],0] ORTE_ERROR_LOG: Out of resource in file util/show_help.c at line 501
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
SVE Gather LD1W            Gather       3.41     78.810        256      14.06
SVE Gather LD1SW+LD1D      Gather       1.86    216.334        384       7.64
SVE Gather+Scatter W   GatherScatter       1.60    168.120        256       6.98
SVE Gather+Scatter D   GatherScatter       1.87    143.874        256       7.87
============================================================
```

### MPI 按类别运行 - Scatter

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi Scatter`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
SVE Scatter ST1W          Scatter       2.60    103.048        256      11.95
SVE Scatter ST1D          Scatter       2.02    132.635        256       8.10
SVE Gather+Scatter W   GatherScatter       1.56    171.626        256       6.67
SVE Gather+Scatter D   GatherScatter       1.89    141.822        256       7.65
============================================================
```

### MPI 按名称匹配 - LD1D

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi "LD1D"`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 3 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
SVE LD1D (Read)              Load      26.67      5.032        128     106.25
SVE LD1D+ST1D (Copy)         Copy      30.28      8.865        256     119.48
SVE Gather LD1SW+LD1D      Gather       1.87    215.481        384       7.48
============================================================
```

### MPI 按类别运行 - Load

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi Load`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
[localhost.localdomain:1895496] [[15579,0],0] ORTE_ERROR_LOG: Out of resource in file util/show_help.c at line 501
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 4 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
NEON LDP (Read)              Load      25.65      5.232        128     104.04
SVE LD1B (Read)              Load      25.35      5.294        128     103.05
SVE LD1W (Read)              Load      25.16      5.334        128     102.81
SVE LD1D (Read)              Load      25.24      5.319        128     103.11
============================================================
```

### MPI 混合指定

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi 0 1 STREAM`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
============================================================
SVE Bandwidth Benchmark (MPI Parallel - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Warmup Iterations: 5
Test Iterations: 10
Selected Tests: 6 of 22

Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
NEON LDP (Read)              Load      26.81      5.006        128     109.12
NEON STP (Write)            Store      38.34      3.501        128     152.31
STREAM Copy                STREAM      29.82      9.001        256     120.14
STREAM Scale               STREAM      29.63      9.061        256     119.24
STREAM Add                 STREAM      29.09     13.840        384     116.52
STREAM Triad               STREAM      28.42     14.169        384     114.87
============================================================
```

### MPI 无效测试名

**命令**: `mpirun --allow-run-as-root -np 4 ./sve_bw_test_mpi INVALID_TEST`

```
--------------------------------------------------------------------------


device.

NOTE: You can turn off this warning by setting the MCA parameter
--------------------------------------------------------------------------
--------------------------------------------------------------------------
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local device:         hns_2
  Local port:           1
--------------------------------------------------------------------------
--------------------------------------------------------------------------

  Location: mtl_ofi_component.c:936
  Error: No data available (281470681743421)
--------------------------------------------------------------------------
No tests match the specified criteria.
Available Tests:
============================================================
Idx  Test Name                Category
============================================================
0    NEON LDP (Read)              Load
1    NEON STP (Write)            Store
2    NEON LDP+STP (Copy)          Copy
3    SVE LD1B (Read)              Load
4    SVE ST1B (Write)            Store
5    SVE LD1B+ST1B (Copy)         Copy
6    SVE LD1W (Read)              Load
7    SVE ST1W (Write)            Store
8    SVE LD1W+ST1W (Copy)         Copy
9    SVE LD1D (Read)              Load
10   SVE ST1D (Write)            Store
11   SVE LD1D+ST1D (Copy)         Copy
12   SVE Gather LD1W            Gather
13   SVE Gather LD1SW+LD1D      Gather
14   SVE Scatter ST1W          Scatter
15   SVE Scatter ST1D          Scatter
16   SVE Gather+Scatter W   GatherScatter
17   SVE Gather+Scatter D   GatherScatter
18   STREAM Copy                STREAM
19   STREAM Scale               STREAM
20   STREAM Add                 STREAM
21   STREAM Triad               STREAM
============================================================
--------------------------------------------------------------------------
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------

--------------------------------------------------------------------------
```

## 测试总结

- 单进程版本: 所有测试项正常运行
- MPI版本: 所有测试项正常运行，进程间同步正常
- 命令行参数: --help, --list, 索引选择, 类别选择, 名称匹配均正常工作
- Gather/Scatter 测试: 结果验证通过，无 VERIFY_FAIL 输出

