# SVE Gather/Scatter 内存带宽测试工具

基于 ARM SVE (Scalable Vector Extension) 的 Gather/Scatter 指令性能测试工具。

## 功能特性

- **Gather 测试**：测试 SVE 向量收集加载指令 (LD1W/LD1SW/LD1D)
- **Scatter 测试**：测试 SVE 向量分散存储指令 (ST1W/ST1D)
- **Gather+Scatter 组合测试**：测试完全非连续内存操作（使用相同索引池）
- **稀疏度控制**：通过稀疏度参数控制访问密度，支持不同测试场景
- **多种索引模式**：支持随机、均匀、去重升序三种索引生成模式
- **随机种子控制**：可配置随机种子，确保测试可重复性
- **索引导出功能**：可导出生成的索引到文件，包含地址偏移信息
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
| `-s, --sparsity <ratio>` | 稀疏度 (0.0-1.0) | 1.0 (100%) |
| `-m, --index-mode <N>` | 索引生成模式 | 0 (随机) |
| `-r, --random-seed <N>` | 随机种子 | 42 |
| `-w, --warmup <N>` | 预热迭代次数 | 5 |
| `-t, --test <N>` | 测试迭代次数 | 10 |
| `-o, --output-indices <file>` | 导出索引到文件 | 无 |
| `-p, --print-all` | 打印所有进程结果 (MPI) | 否 |

### 索引生成模式

| 模式 | 参数值 | 说明 |
|------|--------|------|
| Random | 0 | 完全随机分布（允许重复，无序） |
| Uniform | 1 | 完全均匀分布（固定间隔 stride，无随机） |
| RandomUniqueSorted | 2 | 随机去重后升序排序（缓存友好） |

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
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi

# 显示帮助信息
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi --help

# 使用 Makefile 快捷命令
make run_gs_mpi
```

### MPI 参数控制

```bash
# 参数控制
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi -b 64 -s 0.5

# 自定义随机种子
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi -s 0.01 -r 12345

# 导出索引到文件
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi -s 0.01 -o indices.txt

# 打印所有进程结果
mpirun --allow-run-as-root -np 8 ./gather_scatter_test_mpi -s 0.01 -p

# 运行指定测试项
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi Gather
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi 0 2 4
```

### MPI 版本特点

- 所有进程同步运行相同的测试项
- 使用 MPI_Barrier 在 warmup/test 循环前后同步
- 配置参数通过 MPI_Bcast 广播到所有进程
- 默认仅 rank 0 输出汇总结果（显示 Total(GB/s)）
- 使用 `-p` 参数可打印所有进程的独立结果

## 测试项说明

程序包含 18 个测试项，分为 6 大类：

### 核心测试 (索引 0-5)

| 索引 | 测试名称 | 类别 | 说明 |
|------|----------|------|------|
| 0 | SVE Gather LD1W (Seq-Store) | Gather | 随机地址加载(LD1W) → **顺序存储** |
| 1 | SVE Gather LD1SW+LD1D (Seq-Store) | Gather | 随机地址加载(LD1SW→LD1D) → **顺序存储** |
| 2 | SVE Scatter ST1W (Idx-Store) | Scatter | 顺序加载 → **索引分散存储** (随机地址) |
| 3 | SVE Scatter ST1D (Idx-Store) | Scatter | 顺序加载 → **索引分散存储** (随机地址) |
| 4 | SVE Gather+Scatter W (Idx-Store) | GatherScatter | 随机加载 → **索引分散存储** (完全非连续) |
| 5 | SVE Gather+Scatter D (Idx-Store) | GatherScatter | 随机加载 → **索引分散存储** (完全非连续) |

### Gather 变体测试 (索引 6-11)

测试不同的 Gather 加载模式：

| 索引 | 测试名称 | 说明 |
|------|----------|------|
| 6 | SVE Gather IdxOnly (No-Store) | 仅 Gather 加载索引和数据 |
| 7 | SVE Gather Vec+Idx (No-Store) | 加载向量 + Gather 索引和数据 |
| 8 | SVE Gather Vec+Idx+FMLA (No-Store) | 加载向量 + Gather + FMLA 计算 |
| 9 | SVE Gather Idx+Store (Baseline) | Gather 加载 → 顺序存储 (基准) |
| 10 | SVE Gather Vec+Idx+Store | 加载向量 + Gather + 顺序存储 |
| 11 | SVE Gather Vec+Idx+FMLA+Store | 加载向量 + Gather + FMLA + 顺序存储 |

### Gather D 变体测试 (索引 12-17)

使用 Double 精度的 Gather 变体：

| 索引 | 测试名称 | 说明 |
|------|----------|------|
| 12 | SVE Gather D IdxOnly (No-Store) | 仅 Gather 加载 (Double) |
| 13 | SVE Gather D Vec+Idx (No-Store) | 加载向量 + Gather (Double) |
| 14 | SVE Gather D Vec+Idx+FMLA (No-Store) | 加载向量 + Gather + FMLA (Double) |
| 15 | SVE Gather D Idx+Store (Baseline) | Gather 加载 → 顺序存储 (Double基准) |
| 16 | SVE Gather D Vec+Idx+Store | 加载向量 + Gather + 顺序存储 (Double) |
| 17 | SVE Gather D Vec+Idx+FMLA+Store | 加载向量 + Gather + FMLA + 顺序存储 (Double) |

### Store 特性分类

| 特性 | 测试项 | 说明 |
|------|--------|------|
| **顺序存储 (Seq-Store)** | Gather (0, 1) | dst地址连续递增，硬件预取友好 |
| **索引存储 (Idx-Store)** | Scatter (2, 3), Gather+Scatter (4, 5) | 使用索引向量作为dst地址，真实Scatter语义 |

**关键区别**：
- **Seq-Store**: `st1w [dst, #offset]` - 固定偏移，顺序写入
- **Idx-Store**: `st1w [dst, z_idx.s]` - 向量索引，随机写入

**注意**：Gather+Scatter 测试使用相同的索引池，语义为 `dst[idx[i]] = src[idx[i]]`

## 使用示例

```bash
# 默认参数运行所有测试（100% 稀疏度，随机模式，种子42）
./gather_scatter_test

# 自定义缓冲区和稀疏度
./gather_scatter_test -b 64 -s 0.5

# 100% 稀疏度，完全均匀索引覆盖整个 buffer
./gather_scatter_test -s 1.0 -m 1

# 自定义随机种子
./gather_scatter_test -s 0.5 -r 12345

# 导出索引到文件（包含地址偏移）
./gather_scatter_test -s 0.01 -o indices.txt

# 去重升序模式（缓存友好，测试优化性能）
./gather_scatter_test -s 0.5 -m 2 -b 32

# 小缓冲区快速测试
./gather_scatter_test -b 16 -s 0.01 -w 1 -t 3

# 高精度测试
./gather_scatter_test -w 10 -t 50

# 仅运行 Gather 测试
./gather_scatter_test Gather

# 运行指定索引的测试
./gather_scatter_test 0 2 9 15

# MPI 4 进程测试
mpirun --allow-run-as-root -np 4 ./gather_scatter_test_mpi -s 1.0 -m 1 -b 16

# MPI 8 进程，打印所有进程结果
mpirun --allow-run-as-root -np 8 ./gather_scatter_test_mpi -s 0.01 -b 64 -p
```

## 输出说明

### 单进程版本输出

```
================================================================================
SVE Gather/Scatter Bandwidth Benchmark
================================================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Sparsity: 1.0000 (100.00%)
Index Pool Size: 16777216 elements
Warmup Iterations: 5
Test Iterations: 10
Registered Tests: 18
Random Seed: 42

Index Mode: Random
Max Index: 16777215 (buffer elements: 16777215)
Generated Range: [0, 16777215]
Unique Indices: 10606218 / 16777216 (63.22%)
Coverage: 63.2180% of buffer

Test                                         Category       GB/s   Time(ms)   Data(MB)
================================================================================
SVE Gather LD1W (Seq-Store)                    Gather       1.59    168.970        256
```

新增输出字段：
- **Sparsity**: 稀疏度百分比
- **Random Seed**: 随机种子值
- **Index Mode**: 索引生成模式
- **Max Index**: 最大索引值
- **Generated Range**: 实际生成的索引范围
- **Unique Indices**: 唯一索引数量和比例
- **Coverage**: 索引覆盖 buffer 的比例

### 索导出文件格式

使用 `-o` 参数导出索引时，文件格式如下：

```
# Gather Indices Output
# Index Mode: Random
# Random Seed: 42
# Sparsity: 0.0100
# Index Pool Size: 167772
# Max Index: 16777215
# Generated Range: [0, 16777215]
# Unique Indices: 10606218
# Format: Index | Float_Offset(bytes) | Double_Offset(bytes)
#
  15179413        60717652       121435304
  10336126        41344504        82689008
```

三列数据含义：
- **Index**: 元素索引值
- **Float_Offset**: Float 类型地址偏移 (Index × 4)
- **Double_Offset**: Double 类型地址偏移 (Index × 8)

### MPI 多进程版本输出

#### 默认模式（仅 rank 0）

```
================================================================================
SVE Gather/Scatter Bandwidth Benchmark (MPI - 4 processes)
================================================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 16 MB per array
Sparsity: 1.0000 (100.00%)
Index Pool Size: 2097152 elements
Warmup Iterations: 5
Test Iterations: 10
Registered Tests: 18
Random Seed: 42

Test                                         Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
================================================================================
SVE Gather LD1W (Seq-Store)                    Gather      25.78      1.301         32      85.25
SVE Gather LD1SW+LD1D (Seq-Store)              Gather      32.55      1.031         32     105.69
```

#### 使用 -p 参数（打印所有进程）

```
Test                                         Category       GB/s   Time(ms)   Data(MB)
================================================================================
[Rank 0] SVE Gather LD1W (Seq-Store)           Gather      21.08      0.796         16
[Rank 1] SVE Gather LD1W (Seq-Store)           Gather      21.15      0.793         16
[Rank 2] SVE Gather LD1W (Seq-Store)           Gather      20.12      0.834         16
[Rank 3] SVE Gather LD1W (Seq-Store)           Gather      19.93      0.842         16
```

## 参数选择建议

### 稀疏度 (-s)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 极低稀疏度 | 0.0001-0.001 | 模拟极稀疏访问，索引池很小 |
| 低稀疏度 | 0.01-0.05 | 模拟典型稀疏数据访问 |
| 中等稀疏度 | 0.1-0.5 | 测试较密集的非连续访问 |
| 全覆盖测试 | 1.0 | 索引覆盖整个 buffer（默认值） |

### 索引模式 (-m)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 随机访问模拟 | 0 (Random) | 模拟真实随机数据访问（允许重复） |
| 全范围覆盖 | 1 (Uniform) | 完全均匀分布（固定间隔 stride） |
| 优化访问测试 | 2 (RandomUniqueSorted) | 去重+升序，测试缓存友好性能 |

### 随机种子 (-r)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 默认测试 | 42 | 默认种子，确保可重复性 |
| 自定义测试 | 任意正整数 | 自定义种子，探索不同索引分布 |
| 多次实验 | 42, 100, 12345... | 多个种子，统计性能波动 |

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

1. **Random**：完全随机（允许重复）
```c
srand(random_seed);  // 使用可配置的随机种子
index[i] = rand() % (max_index + 1)
// 特点：真实随机访问，可能有索引冲突，默认种子42
```

2. **Uniform**：完全均匀分布（无随机）
```c
stride = (max_index + 1) / index_pool_size
index[i] = i * stride  // 完全均匀，无随机成分
// 特点：固定间隔，100%覆盖，缓存预取友好
```

3. **RandomUniqueSorted**：去重升序（缓存友好）
```c
srand(random_seed);
// 位图去重
while (unique_count < target && attempts < max_attempts) {
    idx = rand() % (max_index + 1)
    if (!coverage[idx/64] & (1 << (idx%64))) {
        coverage[idx/64] |= (1 << (idx%64))
        unique_indices[unique_count++] = idx
    }
}
// 补充至满足稀疏度（顺序扫描）
for (i = 0; i <= max_index && unique_count < target; i++) {
    if (!covered) unique_indices[unique_count++] = i
}
// 升序排序
qsort(unique_indices, unique_count)
// 特点：100%去重，升序排列，Scatter缓存友好
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

### Store 特性详解

Gather/Scatter 测试的关键区别在于 **Store 操作类型**：

#### 顺序存储 (Seq-Store) - Gather 测试

```asm
// Gather 测试的 Store 指令（顺序存储）
st1w z0.s, p0, [dst, #0, MUL VL]  // dst + 0*VL
st1w z1.s, p0, [dst, #1, MUL VL]  // dst + 1*VL
st1w z2.s, p0, [dst, #2, MUL VL]  // dst + 2*VL
st1w z3.s, p0, [dst, #3, MUL VL]  // dst + 3*VL
add dst, dst, chunk_bytes         // dst指针递增
```

**特点**：
- dst 地址连续递增（固定偏移 #0, #1, #2, #3）
- 硬件预取器可预测访问模式
- 不是真实的 Scatter 语义（仅 Load 是 Gather）

#### 索引存储 (Idx-Store) - Scatter/GatherScatter 测试

```asm
// Scatter/GatherScatter 测试的 Store 指令（索引存储）
ld1w z8.s, p0/z, [idx_ptr, #0, MUL VL]  // 加载索引向量
st1w z0.s, p0, [dst, z8.s, sxtw 2]      // 使用向量索引作为地址
```

**特点**：
- dst 地址由向量索引决定（`z8.s`），完全随机
- 真实的 Scatter 语义（非连续写入）
- 硬件预取器无法预测
- 受索引模式影响：
  - **Mode 0-2**: 随机索引 → 缓存冲突、预取失效
  - **Mode 3**: 升序索引 → 缓存预取友好

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
| SVE Gather LD1W (Seq-Store) | 22.66 | 85.25 | 3.76x |
| SVE Gather LD1SW+LD1D (Seq-Store) | 26.37 | 105.69 | 4.00x |
| SVE Scatter ST1W (Idx-Store) | 19.38 | 69.50 | 3.59x |
| SVE Scatter ST1D (Idx-Store) | 27.20 | 93.34 | 3.44x |
| SVE Gather+Scatter W (Idx-Store) | 16.14 | 56.46 | 3.50x |
| SVE Gather+Scatter D (Idx-Store) | 22.33 | 82.58 | 3.69x |

### 索引模式对比 (16MB buffer, 1% sparsity)

| 模式 | Unique% | Scatter ST1W GB/s | Gather LD1W GB/s | 说明 |
|------|---------|-------------------|------------------|------|
| Random (0) | ~63% | ~4.5 | ~4.6 | 真实随机访问，有重复 |
| Uniform (1) | 100% | ~3.2 | ~3.2 | 完全均匀，固定间隔 |
| RandomUniqueSorted (2) | 100% | ~4.8 | ~3.8 | 升序优化Idx-Store预取 |

**关键观察**：
- **Gather测试（Seq-Store）**：
  - Uniform 模式固定间隔，缓存预取效果最佳
  - Random 模式随机访问，预取器难以预测
- **Scatter测试（Idx-Store）**：
  - RandomUniqueSorted 模式升序索引提升预取效率
  - Uniform 模式固定间隔预取友好
  - **索引模式显著影响带宽**
- **GatherScatter测试（Idx-Store）**：完全非连续访问，带宽最低

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