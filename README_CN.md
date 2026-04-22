# SVE 内存带宽基准测试工具

基于 ARM SVE (Scalable Vector Extension) 和 NEON 指令集的内存带宽性能测试工具。

## 功能特性

- **NEON 指令测试**：测试 LDP/STP 指令的读取、写入和复制带宽
- **SVE 指令测试**：测试 LD1B/LD1W/LD1D 等不同数据宽度的读取、写入和复制带宽
- **STREAM Benchmark**：实现标准 STREAM 测试的四种操作
- **SVE Gather/Scatter 测试**：测试 SVE 向量收集/分散指令的性能
- **MPI 支持**：支持多进程并行测试，汇总总带宽
- **命令行选项**：支持选择性运行指定测试项

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

# 编译所有版本
make all
```

## 运行

### 单进程版本 (sve_bw_test)

```bash
# 运行所有测试（默认）
./sve_bw_test

# 显示帮助信息
./sve_bw_test --help

# 列出所有可用测试项（显示索引号）
./sve_bw_test --list

# 按索引号运行测试
./sve_bw_test 0 1 2          # 运行测试项 0, 1, 2

# 按类别运行测试
./sve_bw_test Load           # 运行所有 Load 类别测试
./sve_bw_test STREAM         # 运行所有 STREAM 类别测试
./sve_bw_test Gather Scatter # 运行 Gather 和 Scatter 类别测试

# 按名称部分匹配运行测试
./sve_bw_test "LD1D"         # 运行所有包含 "LD1D" 的测试
./sve_bw_test "NEON"         # 运行所有包含 "NEON" 的测试
```

### MPI 多进程版本 (sve_bw_test_mpi)

```bash
# 运行所有测试（4进程）
mpirun -np 4 ./sve_bw_test_mpi

# 运行指定测试项
mpirun -np 4 ./sve_bw_test_mpi STREAM
mpirun -np 4 ./sve_bw_test_mpi 0 1 2
mpirun -np 4 ./sve_bw_test_mpi Gather Scatter

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

### SVE Gather 收集加载测试 (索引 12-13)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 12 | SVE Gather LD1W | Gather | SVE 向量收集加载测试，使用 LD1W 指令按索引从随机地址加载32位数据，再顺序存储。测试非连续内存访问性能 |
| 13 | SVE Gather LD1SW+LD1D | Gather | SVE 复合收集加载测试，同时使用 LD1SW (有符号32位扩展到64位) 和 LD1D 指令收集数据 |

### SVE Scatter 分散存储测试 (索引 14-15)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 14 | SVE Scatter ST1W | Scatter | SVE 向量分散存储测试，顺序加载数据后使用 ST1W 指令按索引存储到随机地址。测试非连续内存写入性能 |
| 15 | SVE Scatter ST1D | Scatter | SVE 双字分散存储测试，顺序加载64位数据后使用 ST1D 指令分散存储。使用 LD1SW 指令加载并扩展索引 |

### SVE Gather+Scatter 组合测试 (索引 16-17)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 16 | SVE Gather+Scatter W | GatherScatter | SVE 收集+分散组合测试(字)，使用 LD1W 从随机地址收集数据，使用 ST1W 分散存储到随机地址。模拟完全非连续内存操作 |
| 17 | SVE Gather+Scatter D | GatherScatter | SVE 收集+分散组合测试(双字)，使用 LD1D/LD1SW 收集数据，使用 ST1D 分散存储。使用 LD1SW 指令优化索引加载 |

### STREAM Benchmark 测试 (索引 18-21)

| 索引 | 测试名称 | 类别 | 说明 |
|-----|---------|------|------|
| 18 | STREAM Copy | STREAM | 标准 STREAM Copy 测试：`a[i] = b[i]`，测试内存复制带宽 |
| 19 | STREAM Scale | STREAM | 标准 STREAM Scale 测试：`a[i] = q * b[i]`，测试读取+计算+写入带宽 |
| 20 | STREAM Add | STREAM | 标准 STREAM Add 测试：`a[i] = b[i] + c[i]`，测试双源读取+计算+写入带宽 |
| 21 | STREAM Triad | STREAM | 标准 STREAM Triad 测试：`a[i] = b[i] + q * c[i]`，测试最完整的内存带宽操作 |

## 测试项分类总结

| 类别 | 测试项数量 | 索引范围 | 描述 |
|-----|----------|---------|------|
| Load | 5 | 0, 3, 6, 9 | 顺序内存读取测试 |
| Store | 5 | 1, 4, 7, 10 | 顺序内存写入测试 |
| Copy | 6 | 2, 5, 8, 11, 18 | 顺序内存复制测试 |
| Gather | 2 | 12, 13 | 非连续内存收集加载测试 |
| Scatter | 2 | 14, 15 | 非连续内存分散存储测试 |
| GatherScatter | 2 | 16, 17 | 完全非连续内存操作测试 |
| STREAM | 4 | 18-21 | 标准 STREAM Benchmark |

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

## 许可证

本项目仅供研究和测试使用。