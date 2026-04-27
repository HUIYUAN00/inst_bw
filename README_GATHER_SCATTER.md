# SVE Gather/Scatter 内存带宽测试工具

基于 ARM SVE (Scalable Vector Extension) 的 Gather/Scatter 指令性能测试工具。

## 功能特性

- **Gather 测试**：测试 SVE 向量收集加载指令 (LD1W/LD1SW/LD1D)
- **Scatter 测试**：测试 SVE 向量分散存储指令 (ST1W/ST1D)
- **Gather+Scatter 组合测试**：测试完全非连续内存操作（使用相同索引池）
- **稀疏度控制**：通过稀疏度参数控制访问密度，支持不同测试场景
- **多种索引模式**：支持随机、均匀、热点三种索引生成模式
- **汇编内联循环**：循环逻辑完全内置于汇编中，消除 C 循环开销
- **参数可配置**：缓冲区大小、稀疏度、索引模式、迭代次数均可配置
- **结果验证**：内置结果验证机制，确保测试准确性
- **MPI 支持**：支持多进程并行测试，可选打印所有进程结果

## 编译要求

- ARM 架构处理器，支持 SVE 指令集
- GCC 编译器
- MPI 库（可选，用于并行测试）

## 编译

```bash
# 编译单进程版本
gcc -O3 -march=armv9-a+sve -o gather_scatter_test gather_scatter_test.c

# 编译 MPI 版本
mpicc -O3 -march=armv9-a+sve -DUSE_MPI -o gather_scatter_test_mpi gather_scatter_test.c

# 使用 Makefile
make gather_scatter_test
make gather_scatter_test_mpi
make all
```

## 运行

### 基本用法

```bash
# 运行所有测试（默认参数）
./gather_scatter_test

# 显示帮助信息
./gather_scatter_test --help

# 列出所有测试项
./gather_scatter_test --list
```

### 参数控制

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-b, --buffer-size <MB>` | 缓冲区大小 (MB) | 128 |
| `-s, --sparsity <ratio>` | 稀疏度 (0.0-1.0) | 0.01 (1%) |
| `-m, --index-mode <N>` | 索引生成模式 | 0 (随机) |
| `-w, --warmup <N>` | 预热迭代次数 | 5 |
| `-t, --test <N>` | 测试迭代次数 | 10 |
| `-p, --print-all` | 打印所有进程结果 (MPI) | 否 |

### 索引生成模式

| 模式 | 参数值 | 说明 |
|------|--------|------|
| Random | 0 | 完全随机分布 |
| Uniform | 1 | 均匀覆盖整个 buffer 范围 |
| Hotspot | 2 | 80% 访问集中在 10% 区域 |

### 测试选择

| 方式 | 说明 |
|------|------|
| `<index>` | 按索引号选择测试 (0-5) |
| `<name>` | 按名称部分匹配 |
| `<category>` | 按类别选择 (Gather/Scatter/GatherScatter) |

## MPI 多进程版本

### 基本用法

```bash
# 运行 4 进程测试（推荐使用此参数避免警告）
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi

# 显示帮助信息
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi --help

# 使用 Makefile 快捷命令
make run_gs_mpi
```

### MPI 参数控制

```bash
# 参数控制
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi -b 64 -s 0.5

# 热点模式测试
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi -s 0.5 -m 2

# 打印所有进程结果
mpirun --mca btl ^openib --mca mtl ^ofi -np 8 ./gather_scatter_test_mpi -s 0.01 -p

# 运行指定测试项
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi Gather
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi 0 2 4
```

### MPI 版本特点

- 所有进程同步运行相同的测试项
- 使用 MPI_Barrier 在 warmup/test 循环前后同步
- 配置参数通过 MPI_Bcast 广播到所有进程
- 默认仅 rank 0 输出汇总结果（显示 Total(GB/s)）
- 使用 `-p` 参数可打印所有进程的独立结果

## 测试项说明

| 索引 | 测试名称 | 类别 | 说明 |
|------|----------|------|------|
| 0 | SVE Gather LD1W | Gather | 使用 LD1W 指令按索引从随机地址加载 32 位数据，顺序存储 |
| 1 | SVE Gather LD1SW+LD1D | Gather | 使用 LD1SW 加载有符号 32 位扩展到 64 位，配合 LD1D 收集数据 |
| 2 | SVE Scatter ST1W | Scatter | 顺序加载 32 位数据，使用 ST1W 按索引分散存储到随机地址 |
| 3 | SVE Scatter ST1D | Scatter | 顺序加载 64 位数据，使用 ST1D 分散存储，配合 LD1SW 加载索引 |
| 4 | SVE Gather+Scatter W | GatherScatter | 完全非连续：使用相同索引池进行 LD1W 收集 + ST1W 分散 (32 位) |
| 5 | SVE Gather+Scatter D | GatherScatter | 完全非连续：使用相同索引池进行 LD1D 收集 + ST1D 分散 (64 位) |

**注意**：Gather+Scatter 测试使用相同的索引池，语义为 `dst[idx[i]] = src[idx[i]]`

## 使用示例

```bash
# 默认参数运行所有测试（1% 稀疏度，随机模式）
./gather_scatter_test

# 自定义缓冲区和稀疏度
./gather_scatter_test -b 64 -s 0.5

# 100% 稀疏度，均匀索引覆盖整个 buffer
./gather_scatter_test -s 1.0 -m 1

# 热点模式（80%访问集中在10%区域）
./gather_scatter_test -s 0.5 -m 2 -b 32

# 小缓冲区快速测试
./gather_scatter_test -b 16 -s 0.01 -w 1 -t 3

# 高精度测试
./gather_scatter_test -w 10 -t 50

# MPI 4 进程测试
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi -s 1.0 -m 1 -b 16

# MPI 8 进程，打印所有进程结果
mpirun --mca btl ^openib --mca mtl ^ofi -np 8 ./gather_scatter_test_mpi -s 0.01 -b 64 -p
```

## 输出说明

### 单进程版本输出

```
============================================================
SVE Gather/Scatter Bandwidth Benchmark
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Sparsity: 0.0100 (1.00%)
Index Pool Size: 1048576 elements
Warmup Iterations: 5
Test Iterations: 10
Registered Tests: 6

Index Mode: Random
Max Index: 16777215 (buffer elements: 16777215)
Generated Range: [26, 16777208]
Unique Indices: 10444 / 10485 (99.61%)
Coverage: 0.9960% of buffer

Test                          Category       GB/s   Time(ms)   Data(MB)
============================================================
SVE Gather LD1W                 Gather      11.41      1.470         16
```

新增输出字段：
- **Sparsity**: 稀疏度百分比
- **Index Mode**: 索引生成模式
- **Max Index**: 最大索引值
- **Generated Range**: 实际生成的索引范围
- **Unique Indices**: 唯一索引数量和比例
- **Coverage**: 索引覆盖 buffer 的比例

### MPI 多进程版本输出

#### 默认模式（仅 rank 0）

```
============================================================
SVE Gather/Scatter Bandwidth Benchmark (MPI - 4 processes)
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 16 MB per array
Sparsity: 1.0000 (100.00%)
Index Pool Size: 2097152 elements

Test                          Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
============================================================
SVE Gather LD1W                 Gather      25.78      1.301         32      85.25
SVE Gather LD1SW+LD1D           Gather      32.55      1.031         32     105.69
```

#### 使用 -p 参数（打印所有进程）

```
Test                          Category       GB/s   Time(ms)   Data(MB)
============================================================
[Rank 0] SVE Gather LD1W                 Gather      21.08      0.796         16
[Rank 1] SVE Gather LD1W                 Gather      21.15      0.793         16
[Rank 2] SVE Gather LD1W                 Gather      20.12      0.834         16
[Rank 3] SVE Gather LD1W                 Gather      19.93      0.842         16
```

## 参数选择建议

### 稀疏度 (-s)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 极低稀疏度 | 0.0001-0.001 | 模拟极稀疏访问，索引池很小 |
| 低稀疏度 | 0.01-0.05 | 模拟典型稀疏数据访问 |
| 中等稀疏度 | 0.1-0.5 | 测试较密集的非连续访问 |
| 全覆盖测试 | 1.0 | 索引覆盖整个 buffer |

### 索引模式 (-m)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 随机访问模拟 | 0 (Random) | 模拟随机数据访问模式 |
| 全范围覆盖 | 1 (Uniform) | 均匀索引，覆盖整个 buffer |
| 热点数据 | 2 (Hotspot) | 模拟热点数据访问（80%集中） |

### 缓冲区大小 (-b)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 快速测试 | 8-16 | 快速验证功能 |
| 标准测试 | 32-128 | 平衡测试时间和准确度 |
| 大内存测试 | 256-512 | 测试大容量内存性能 |

### 迭代次数 (-w/-t)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 快速验证 | -w 1 -t 3 | 快速功能验证 |
| 标准测试 | -w 5 -t 10 | 默认设置 |
| 高精度 | -w 10 -t 50 | 减少测量波动 |

## 技术细节

### 稀疏度计算

索引池大小根据稀疏度动态计算：

```c
index_pool_size = sparsity * (buffer_size / sizeof(int64_t))
```

例如：
- Buffer Size: 128MB = 16M 个 int64_t 元素
- Sparsity: 0.01 → index_pool_size = 160K 个索引
- Sparsity: 1.0 → index_pool_size = 16M 个索引（全覆盖）

### 索引生成算法

三种索引生成模式：

1. **Random**：完全随机
```c
index[i] = rand() % (max_index + 1)
```

2. **Uniform**：均匀分布
```c
stride = (max_index + 1) / index_pool_size
index[i] = i * stride + rand() % stride
```

3. **Hotspot**：热点模式
```c
hotspot_size = max_index / 10
hotspot_start = rand() % (max_index - hotspot_size)
if (rand() % 100 < 80)  // 80%概率
    index[i] = hotspot_start + rand() % hotspot_size
else
    index[i] = rand() % (max_index + 1)
```

### 汇编内联循环

循环逻辑完全内置于汇编中，使用 ARM64 分支指令：

```asm
mov x16, iterations        // 循环计数器
mov x17, #0                // 重置计数器（初始为0触发重置）
1:
    cmp x17, #0
    b.ne 2f
    mov x20, idx_base      // 重置索引指针
    mov x17, reset_value   // 设置重置计数
2:
    // ... SVE Gather/Scatter 指令 ...
    add x20, x20, increment
    subs x17, x17, #1
    subs x16, x16, #1
    b.ne 1b                // 继续循环
```

### ld1sw 指令优化

在 64 位操作中使用 `ld1sw` 替代 `ld1w + sunpklo`：
- 直接加载 32 位有符号整数并扩展为 64 位
- 减少指令数量，提高效率

### 结果验证

所有 Gather/Scatter 测试包含结果验证：
- Gather: 验证收集的数据是否与源数据匹配
- Scatter: 统计每个位置的写入次数，验证正确性
- Gather+Scatter: 双向验证（使用相同索引池）

验证失败时会输出 `VERIFY_FAIL(n)` 标记。

## 性能对比

### 单进程 vs MPI 4进程 (16MB buffer, 100% sparsity, uniform)

| 测试项 | 单进程 GB/s | MPI 4进程 总带宽 GB/s | 增倍比 |
|-------|------------|---------------------|-------|
| SVE Gather LD1W | 22.66 | 85.25 | 3.76x |
| SVE Gather LD1SW+LD1D | 26.37 | 105.69 | 4.00x |
| SVE Scatter ST1W | 19.38 | 69.50 | 3.59x |
| SVE Scatter ST1D | 27.20 | 93.34 | 3.44x |
| SVE Gather+Scatter W | 16.14 | 56.46 | 3.50x |
| SVE Gather+Scatter D | 22.33 | 82.58 | 3.69x |

## 清理

```bash
make clean
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `gather_scatter_test.c` | 源代码 |
| `Makefile` | 编译配置 |
| `README_GATHER_SCATTER.md` | 本文档 |

## 许可证

本项目仅供研究和测试使用。