# SVE 内存带宽基准测试工具

基于 ARM SVE (Scalable Vector Extension) 和 NEON 指令集的内存带宽性能测试工具。

## 功能特性

- **NEON 指令测试**：测试 LDP/STP 指令的读取、写入和复制带宽
- **SVE 指令测试**：测试 LD1B/LD1W/LD1D 等不同数据宽度的读取、写入和复制带宽
- **参数可配置**：缓冲区大小、迭代次数可通过命令行控制
- **MPI 支持**：支持多进程并行测试，汇总总带宽
- **命令行选项**：支持选择性运行指定测试项

**注意：** Gather/Scatter 和 STREAM 测试已移至独立程序 `gather_scatter_test`，详见 `README_GATHER_SCATTER.md`。

## 编译要求

- ARM 架构处理器，支持 SVE 指令集
- GCC 编译器
- MPI 库（可选，用于并行测试）

## 编译

```bash
# 编译单进程版本
make sve_bw_test

# 编译 MPI 版本
make sve_bw_test_mpi

# 编译 Gather/Scatter 独立版本
make gather_scatter_test

# 编译所有版本
make all
```

## 运行

### 单进程版本 (sve_bw_test)

```bash
# 运行所有测试（默认参数）
./sve_bw_test

# 参数控制
./sve_bw_test -b 64                         # 64MB 缓冲区
./sve_bw_test -b 32 -w 2 -t 5               # 32MB 缓冲区，2次预热，5次测试
./sve_bw_test -b 256 Copy                   # 256MB 缓冲区，运行 Copy 测试

# 显示帮助信息
./sve_bw_test --help

# 列出所有可用测试项
./sve_bw_test --list

# 按索引号运行测试
./sve_bw_test 0 6 9          # 运行测试项 0, 6, 9

# 按类别运行测试
./sve_bw_test Load           # 运行所有 Load 类别测试
./sve_bw_test Store          # 运行所有 Store 类别测试
./sve_bw_test Copy           # 运行所有 Copy 类别测试

# 按名称部分匹配运行测试
./sve_bw_test "LD1D"         # 运行所有包含 "LD1D" 的测试
./sve_bw_test "NEON"         # 运行所有包含 "NEON" 的测试
```

**sve_bw_test 参数说明：**

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `-b <MB>` | 缓冲区大小 | 128 MB |
| `-w <N>` | 预热迭代次数 | 5 |
| `-t <N>` | 测试迭代次数 | 10 |

### MPI 多进程版本 (sve_bw_test_mpi)

```bash
# 运行所有测试（4进程）
mpirun -np 4 ./sve_bw_test_mpi

# 参数控制
mpirun -np 4 ./sve_bw_test_mpi -b 64
mpirun -np 4 ./sve_bw_test_mpi -b 32 -w 3 -t 10

# 运行指定测试项
mpirun -np 4 ./sve_bw_test_mpi Load
mpirun -np 4 ./sve_bw_test_mpi 0 6 9
mpirun -np 4 ./sve_bw_test_mpi Copy

# 显示帮助或列表
mpirun -np 4 ./sve_bw_test_mpi --help
mpirun -np 4 ./sve_bw_test_mpi --list

# 使用 Makefile 快捷命令
make run_single  # 单进程运行所有测试
make run         # MPI 4进程运行所有测试
```

**MPI 版本特点：**
- 所有进程同步运行相同的测试项
- 命令行参数由 rank 0 进程解析后广播到所有进程
- 输出显示每个进程的带宽和所有进程的总带宽

### Gather/Scatter 独立版本 (gather_scatter_test)

```bash
# 运行所有测试（默认参数）
./gather_scatter_test

# 参数控制
./gather_scatter_test -b 64 -i 512          # 64MB 缓冲区，512K 下标池
./gather_scatter_test -w 10 -t 50           # 10次预热，50次测试迭代
./gather_scatter_test -b 256 Gather         # 256MB 缓冲区，运行 Gather 测试

# 显示帮助
./gather_scatter_test --help
./gather_scatter_test --list

# 使用 Makefile 快捷命令
make run_gs    # 运行 gather_scatter_test
```

**Gather/Scatter 版本特点：**
- 缓冲区大小、下标池大小可通过命令行参数配置
- 支持按索引、名称、类别选择测试项
- 内置结果验证机制
- 详细文档见 `README_GATHER_SCATTER.md`

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `-b <MB>` | 缓冲区大小 | 128 MB |
| `-i <K>` | 下标池大小 | 1024 K |
| `-w <N>` | 预热迭代次数 | 5 |
| `-t <N>` | 测试迭代次数 | 10 |

## 测试项目详解

### NEON 指令测试 (索引 0-2)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 0 | NEON LDP (Read) | Load | NEON 成对加载指令 (LDP) 顺序读取测试，测试内存读取带宽 |
| 1 | NEON STP (Write) | Store | NEON 成对存储指令 (STP) 顺序写入测试，测试内存写入带宽 |
| 2 | NEON LDP+STP (Copy) | Copy | NEON 加载+存储组合测试，模拟内存复制操作 |

### SVE 字节操作测试 (索引 3-5)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 3 | SVE LD1B (Read) | Load | SVE 字节加载指令 (LD1B) 顺序读取测试 |
| 4 | SVE ST1B (Write) | Store | SVE 字节存储指令 (ST1B) 顺序写入测试 |
| 5 | SVE LD1B+ST1B (Copy) | Copy | SVE 字节加载+存储组合测试，模拟字节级内存复制 |

### SVE 字操作测试 (索引 6-8)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 6 | SVE LD1W (Read) | Load | SVE 字加载指令 (LD1W) 顺序读取测试，32位数据宽度 |
| 7 | SVE ST1W (Write) | Store | SVE 字存储指令 (ST1W) 顺序写入测试，32位数据宽度 |
| 8 | SVE LD1W+ST1W (Copy) | Copy | SVE 字加载+存储组合测试，模拟32位内存复制 |

### SVE 双字操作测试 (索引 9-11)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 9 | SVE LD1D (Read) | Load | SVE 双字加载指令 (LD1D) 顺序读取测试，64位数据宽度 |
| 10 | SVE ST1D (Write) | Store | SVE 双字存储指令 (ST1D) 顺序写入测试，64位数据宽度 |
| 11 | SVE LD1D+ST1D (Copy) | Copy | SVE 双字加载+存储组合测试，模拟64位内存复制 |

## 测试项分类总结

| 类别 | 测试项数量 | 索引范围 | 描述 |
|-----|----------|---------|------|
| Load | 4 | 0, 3, 6, 9 | 顺序内存读取测试 |
| Store | 4 | 1, 4, 7, 10 | 顺序内存写入测试 |
| Copy | 4 | 2, 5, 8, 11 | 顺序内存复制测试 |

## 输出说明

### 单进程版本输出

```
Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      28.56      4.699        128
```

- **Test**: 测试项名称
- **Category**: 测试类别
- **GB/s**: 单进程带宽（吉字节/秒）
- **Time(ms)**: 单次测试执行时间（毫秒）
- **Data(MB)**: 单次测试处理的数据量（兆字节）

### MPI 多进程版本输出

```
Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
NEON LDP (Read)              Load      28.56      4.699        128      114.24
```

额外显示 **Total(GB/s)**：所有进程的总带宽之和

## 清理

```bash
make clean
```
Test                     Category       GB/s   Time(ms)   Data(MB)
============================================================
NEON LDP (Read)              Load      28.56      4.699        128
```

- **Test**: 测试项名称
- **Category**: 测试类别
- **GB/s**: 单进程带宽（吉字节/秒）
- **Time(ms)**: 单次测试执行时间（毫秒）
- **Data(MB)**: 单次测试处理的数据量（兆字节）

### MPI 多进程版本输出

```
Test                     Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
NEON LDP (Read)              Load      28.56      4.699        128      114.24
```

额外显示 **Total(GB/s)**：所有进程的总带宽之和

## 配置参数

可在源代码中修改以下参数：

- `BUFFER_SIZE`：测试缓冲区大小（默认 128MB）
- `WARMUP_ITER`：预热迭代次数（默认 5 次）
- `TEST_ITER`：测试迭代次数（默认 10 次）
- `INDEX_POOL_SIZE`：Gather/Scatter 测试的索引池大小（默认 1M）

## 技术细节

### ld1sw 指令优化

在涉及64位操作的 Gather/Scatter 测试中，使用 `ld1sw` 指令替代 `ld1w + sunpklo` 组合：
- `ld1sw zD.d, p/z, [base]` 直接加载32位有符号整数并扩展为64位
- 减少指令数量，提高效率

### 测试验证

Gather 和 Scatter 相关测试包含结果验证，验证失败时会输出 `VERIFY_FAIL(n)` 标记。

## 清理

```bash
make clean
```

## 相关文档

| 文档 | 说明 |
|------|------|
| `README_CN.md` | 主文档 |
| `README_GATHER_SCATTER.md` | Gather/Scatter 独立版本详细文档 |
| `test_results.md` | 测试结果记录 |

## 许可证

本项目仅供研究和测试使用。