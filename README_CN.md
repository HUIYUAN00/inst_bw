# SVE 内存带宽基准测试工具

基于 ARM SVE (Scalable Vector Extension) 和 NEON 指令集的内存带宽性能测试工具。

## 功能特性

- **NEON 指令测试**：测试 LDP/STP 指令的读取、写入和复制带宽
- **SVE 指令测试**：测试 LD1B/LD1W/LD1D 等不同数据宽度的读取、写入和复制带宽
- **STREAM Benchmark**：实现标准 STREAM 测试的四种操作
  - Copy: `a[i] = b[i]`
  - Scale: `a[i] = q * b[i]`
  - Add: `a[i] = b[i] + c[i]`
  - Triad: `a[i] = b[i] + q * c[i]`
- **SVE Gather 测试**：测试 SVE 向量收集指令的性能
- **MPI 支持**：支持多进程并行测试，汇总总带宽

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

```bash
# 单进程测试（运行所有测试）
./sve_bw_test

# 显示帮助信息
./sve_bw_test --help

# 列出所有可用测试项
./sve_bw_test --list

# 运行指定测试（按索引）
./sve_bw_test 0 1 2

# 运行指定测试（按类别）
./sve_bw_test STREAM

# 运行指定测试（按名称部分匹配）
./sve_bw_test "LD1D"

# MPI 并行测试（4进程）
mpirun -np 4 ./sve_bw_test_mpi

# MPI 并行测试指定测试项
mpirun -np 4 ./sve_bw_test_mpi Gather Scatter

# 或使用 Makefile 快捷命令
make run_single  # 单进程
make run         # MPI 4进程
```

## 测试项目

| 测试名称 | 类别 | 说明 |
|---------|------|------|
| NEON LDP (Read) | Load | NEON 成对加载读取测试 |
| NEON STP (Write) | Store | NEON 成对存储写入测试 |
| NEON LDP+STP (Copy) | Copy | NEON 加载+存储复制测试 |
| SVE LD1B (Read) | Load | SVE 字节读取测试 |
| SVE ST1B (Write) | Store | SVE 字节写入测试 |
| SVE LD1B+ST1B (Copy) | Copy | SVE 字节复制测试 |
| SVE LD1W (Read) | Load | SVE 字读取测试 |
| SVE ST1W (Write) | Store | SVE 字写入测试 |
| SVE LD1W+ST1W (Copy) | Copy | SVE 字复制测试 |
| SVE LD1D (Read) | Load | SVE 双字读取测试 |
| SVE ST1D (Write) | Store | SVE 双字写入测试 |
| SVE LD1D+ST1D (Copy) | Copy | SVE 双字复制测试 |
| SVE Gather LD1W | Gather | SVE 收集加载测试（字） |
| SVE Gather LD1SW+LD1D | Gather | SVE 收集加载测试（双字） |
| SVE Scatter ST1W | Scatter | SVE 分散存储测试（字） |
| SVE Scatter ST1D | Scatter | SVE 分散存储测试（双字） |
| SVE Gather+Scatter W | GatherScatter | SVE 收集加载+分散存储测试（字） |
| SVE Gather+Scatter D | GatherScatter | SVE 收集加载+分散存储测试（双字） |
| STREAM Copy | STREAM | STREAM Copy 测试 |
| STREAM Scale | STREAM | STREAM Scale 测试 |
| STREAM Add | STREAM | STREAM Add 测试 |
| STREAM Triad | STREAM | STREAM Triad 测试 |

## 输出说明

程序输出包含以下信息：
- SVE 向量长度
- 缓冲区大小
- 各测试项的带宽（GB/s）、执行时间、数据量
- MPI 模式下额外显示所有进程的总带宽

## 配置参数

可在源代码中修改以下参数：
- `BUFFER_SIZE`：测试缓冲区大小（默认 128MB）
- `WARMUP_ITER`：预热迭代次数（默认 5 次）
- `TEST_ITER`：测试迭代次数（默认 10 次）

## 清理

```bash
make clean
```

## 许可证

本项目仅供研究和测试使用。