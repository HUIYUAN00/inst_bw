# inst_bw 项目总结 - 大模型快速理解文档

## 项目概述

**项目名称**: inst_bw (Instruction Bandwidth)
**项目地址**: https://github.com/HUIYUAN00/inst_bw
**核心功能**: ARM SVE 指令带宽性能测试工具

**主要测试工具**:
1. `sve_bw_test.c` - SVE连续内存带宽测试（STREAM风格）
2. `gather_scatter_test.c` - SVE Gather/Scatter非连续内存带宽测试

---

## gather_scatter_test.c 详细说明

### 功能定位

测试ARM SVE的Gather（收集加载）和Scatter（分散存储）指令性能，用于评估非连续内存访问的带宽。

### 测试架构

#### 测试项（共6项）

| Idx | 测试名称 | 类别 | Gather指令 | Scatter指令 | 数据类型 |
|-----|---------|------|-----------|------------|---------|
| 0 | SVE Gather LD1W | Gather | ld1w (gather) | st1w (连续) | float (32-bit) |
| 1 | SVE Gather LD1SW+LD1D | Gather | ld1d (gather) | st1d (连续) | double (64-bit) |
| 2 | SVE Scatter ST1W | Scatter | ld1w (连续) | st1w (scatter) | float |
| 3 | SVE Scatter ST1D | Scatter | ld1d (连续) | st1d (scatter) | double |
| 4 | SVE Gather+Scatter W | GatherScatter | ld1w (gather) | st1w (scatter) | float |
| 5 | SVE Gather+Scatter D | GatherScatter | ld1d (gather) | st1d (scatter) | double |

#### Gather+Scatter特性

- **相同索引池**: Gather和Scatter使用相同的索引数组
- **语义**: `dst[indices[i]] = src[indices[i]]` (原地拷贝)
- **索引向量**: 一次加载8组索引向量（z8-z15），用于Gather和Scatter

### 核心参数

#### 稀疏度参数 (sparsity)

```c
// 定义
static double sparsity = 0.01;  // 默认1%

// 计算
index_pool_size = sparsity * (buffer_size / sizeof(int64_t));
```

**含义**:
- 稀疏度 = 索引数量 / buffer元素总数
- 0.01表示1%的元素被访问
- 1.0表示100%全覆盖

#### 索引生成模式 (index_mode)

| 模式 | 值 | 说明 | 适用场景 |
|------|---|------|---------|
| Random | 0 | 完全随机分布 | 模拟随机访问 |
| Uniform | 1 | 均匀覆盖整个buffer | 测试全范围性能 |
| Hotspot | 2 | 80%集中在10%区域 | 模拟热点数据 |

**Uniform算法细节**:
```c
stride = (max_idx + 1) / index_pool_size;
remainder = (max_idx + 1) - stride * index_pool_size;
base = i * stride + (i < remainder ? i : remainder);
idx = base + rand() % stride;
```

**Hotspot算法**:
```c
hotspot_size = max_idx / 10;
hotspot_start = rand() % (max_idx - hotspot_size);
if (rand() % 100 < 80)  // 80%概率访问热点
    idx = hotspot_start + rand() % hotspot_size;
else
    idx = rand() % max_idx;
```

#### 其他参数

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| buffer_size | -b | 128MB | 每个数组大小 |
| warmup_iter | -w | 5 | 预热迭代次数 |
| test_iter | -t | 10 | 测试迭代次数 |
| print_all_ranks | -p | false | MPI打印所有进程结果 |

### 汇编实现核心逻辑

#### 循环内联架构

所有测试函数将循环逻辑完全内置于汇编中，消除C循环开销：

```asm
// 寄存器分配
mov x16, iterations        // 总迭代次数
mov x17, #0                // 索引重置计数器（初始为0触发重置）
mov x18, gather_indices    // 索引池基址
mov x19, increment         // 索引指针增量
mov x20, idx_base          // 当前索引指针

// 循环结构
1:
    cmp x17, #0            // 检查是否需要重置
    b.ne 2f                // 不需要则跳到执行
    mov x20, x18            // 重置索引指针
    mov x17, reset_value   // 设置重置计数
    
2:
    // SVE Gather/Scatter指令执行
    ld1w z8.s, p0/z, [x20, #0, MUL VL]  // 加载索引向量
    ld1w z0.s, p0/z, [src, z8.s, sxtw 2]  // Gather加载
    st1w z0.s, p0, [dst, z8.s, sxtw 2]  // Scatter存储
    
    add x20, x20, x19      // 指针前进
    subs x17, x17, #1      // 重置计数递减
    subs x16, x16, #1      // 总迭代递减
    b.ne 1b                // 继续循环
```

#### SVE指令详解

**Gather加载**:
```asm
ld1w z0.s, p0/z, [src, z8.s, sxtw 2]
// z0.s = src[z8.s[i] * 4] (每个索引乘4，float字节偏移)

ld1d z0.d, p0/z, [src, z8.d, lsl 3]
// z0.d = src[z8.d[i] * 8] (每个索引乘8，double字节偏移)
```

**Scatter存储**:
```asm
st1w z0.s, p0, [dst, z8.s, sxtw 2]
// dst[z8.s[i] * 4] = z0.s[i]

st1d z0.d, p0, [dst, z8.d, lsl 3]
// dst[z8.d[i] * 8] = z0.d[i]
```

**索引扩展**:
```asm
ld1sw z4.d, p0/z, [idx, #0, MUL VL]
// 加载32-bit有符号整数，扩展为64-bit
// 用于64位数据的Gather/Scatter
```

#### 批量处理

每轮循环处理8个向量（VL长度）的数据：
- **float测试**: 8 VL * 32-bit = 8 VL个元素
- **double测试**: 4 VL * 64-bit = 4 VL个元素

**索引重置机制**:
```c
idx_pool_iters = index_pool_size / (vl * 8);  // 每多少轮重置
x17计数器：每轮递减，到0时重置索引指针
```

### MPI并行测试

#### 编译与运行

```bash
# 编译
mpicc -O3 -march=armv9-a+sve -DUSE_MPI -o gather_scatter_test_mpi gather_scatter_test.c

# 运行（推荐参数避免警告）
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi

# 打印所有进程
mpirun --mca btl ^openib --mca mtl ^ofi -np 8 ./gather_scatter_test_mpi -p
```

#### MPI架构

```c
// 参数广播
MPI_Bcast(&buffer_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&index_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&print_all_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);

// 同步
MPI_Barrier(MPI_COMM_WORLD);  // warmup前后、测试前后

// 结果汇总
MPI_Reduce(&bandwidth, &total_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

#### 输出模式

**默认模式（仅rank 0）**:
```
Test                          Category       GB/s   Time(ms)   Data(MB) Total(GB/s)
SVE Gather LD1W                 Gather      25.78      1.301         32      85.25
```

**-p 模式（所有进程）**:
```
[Rank 0] SVE Gather LD1W                 Gather      21.08      0.796         16
[Rank 1] SVE Gather LD1W                 Gather      21.15      0.793         16
[Rank 2] SVE Gather LD1W                 Gather      20.12      0.834         16
[Rank 3] SVE Gather LD1W                 Gather      19.93      0.842         16
```

### 性能数据

#### 单进程典型结果 (16MB, 100% sparsity, Uniform)

| 测试项 | GB/s | 时间(ms) |
|-------|------|---------|
| SVE Gather LD1W | 22-26 | 0.7-1.5 |
| SVE Gather LD1SW+LD1D | 26-32 | 0.6-1.3 |
| SVE Scatter ST1W | 19-20 | 0.9-1.7 |
| SVE Scatter ST1D | 27-28 | 0.6-1.2 |
| SVE Gather+Scatter W | 16-17 | 1.0-2.1 |
| SVE Gather+Scatter D | 22-23 | 0.7-1.5 |

#### MPI多进程扩展性 (4进程)

| 测试项 | 单进程 | 4进程总带宽 | 增倍比 |
|-------|--------|------------|-------|
| SVE Gather LD1W | 22.66 | 85.25 | 3.76x |
| SVE Gather LD1SW+LD1D | 26.37 | 105.69 | 4.00x |
| SVE Scatter ST1W | 19.38 | 69.50 | 3.59x |
| SVE Scatter ST1D | 27.20 | 93.34 | 3.44x |
| SVE Gather+Scatter W | 16.14 | 56.46 | 3.50x |
| SVE Gather+Scatter D | 22.33 | 82.58 | 3.69x |

#### 稀疏度影响

| 稀疏度 | 覆盖率 | Gather LD1W GB/s | Scatter ST1W GB/s |
|--------|--------|-----------------|------------------|
| 100% | 100% | 26.03 | 19.30 |
| 50% | 50% | 19.83 | 16.23 |
| 10% | 10% | 5.60 | 4.87 |
| 1% | 1% | 3.85 | 4.00 |
| 0.1% | 0.1% | ~2-3 | ~3-4 |

#### 索引模式影响

| 模式 | 覆盖率 | 带宽影响 |
|------|--------|---------|
| Uniform | 理论值 | 基准性能 |
| Random | ~理论值 | 稍低于Uniform |
| Hotspot | 18% (50%稀疏度) | 最低（热点冲突） |

### 使用示例库

#### 基本测试

```bash
# 默认参数（1%稀疏度，随机模式）
./gather_scatter_test

# 全覆盖均匀分布
./gather_scatter_test -s 1.0 -m 1

# 热点模式
./gather_scatter_test -s 0.5 -m 2 -b 32

# 高精度测试
./gather_scatter_test -w 10 -t 50

# 快速测试
./gather_scatter_test -b 8 -w 1 -t 3
```

#### MPI测试

```bash
# 4进程基本测试
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi

# 8进程打印所有结果
mpirun --mca btl ^openib --mca mtl ^ofi -np 8 ./gather_scatter_test_mpi -s 0.01 -p

# 大buffer测试
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi -b 256 -s 1.0 -m 1

# 热点模式
mpirun --mca btl ^openib --mca mtl ^ofi -np 4 ./gather_scatter_test_mpi -s 0.3 -m 2
```

#### 参数组合

```bash
# 低稀疏度 + 大buffer
./gather_scatter_test -s 0.001 -b 512

# 中等稀疏度 + 均匀模式
./gather_scatter_test -s 0.5 -m 1 -b 128

# 极低稀疏度（随机模式）
./gather_scatter_test -s 0.0001 -m 0 -w 3 -t 20

# 特定测试项
./gather_scatter_test Gather
./gather_scatter_test 0 4
./gather_scatter_test Scatter
```

### 输出解读

#### 关键输出字段

```
Index Mode: Uniform
Max Index: 2097151 (buffer elements: 2097151)
Generated Range: [0, 2097151]
Unique Indices: 2097152 / 2097152 (100.00%)
Coverage: 100.0000% of buffer
```

**含义**:
- **Max Index**: buffer最大元素索引
- **Generated Range**: 实际生成的索引范围
- **Unique Indices**: 唯一索引数（访问了多少不同位置）
- **Coverage**: 索引覆盖buffer的比例（=稀疏度）

#### 带宽计算

```c
bytes_per_iter = buffer_size * 2;  // 读+写各1次
bandwidth = bytes / time_sec / 1e9;  // GB/s
```

#### 验证标记

```
SVE Gather LD1W                 Gather      11.41      1.470         16
SVE Gather LD1W                 Gather       3.81      4.404         16  VERIFY_FAIL(5)
```

- **无标记**: 验证通过
- **VERIFY_FAIL(n)**: 有n个错误

### 技术要点

#### ld1sw优化

在64位操作中使用`ld1sw`替代`ld1w + sunpklo`：
- 一次指令完成32-bit加载+64-bit扩展
- 减少指令数量，提高效率

#### 索引字节偏移

- **sxtw 2**: 索引值乘4（float 32-bit = 4字节）
- **lsl 3**: 索引值乘8（double 64-bit = 8字节）

#### 结果验证机制

```c
// Gather验证
dst[i]应该等于src[indices[i]]

// Scatter验证
统计每个位置的写入次数，验证值正确性

// Gather+Scatter验证
双向验证：dst[indices[i]] = src[indices[i]]
```

### 代码结构

#### 主要函数

```c
// 测试函数（汇编内联）
sve_gather_ld1w_ld1w()       // 测试0
sve_gather_ld1sw_ld1d()      // 测试1
sve_scatter_st1w()           // 测试2
sve_scatter_st1d()           // 测试3
sve_gather_scatter_w()       // 测试4
sve_gather_scatter_d()       // 测试5

// 验证函数
verify_gather_ld1w_ld1w()
verify_gather_ld1sw_ld1d()
verify_scatter_st1w()
verify_scatter_st1d()
verify_gather_scatter_w()
verify_gather_scatter_d()

// 辅助函数
get_bandwidth()
should_run_test()
print_usage()
print_tests()
```

#### 全局变量

```c
static int warmup_iter = 5;
static int test_iter = 10;
static uint64_t buffer_size = 128 * 1024 * 1024;
static double sparsity = 0.01;
static int index_mode = 0;
static int print_all_ranks = 0;
static uint64_t index_pool_size = 0;
static int32_t *gather_indices = NULL;
```

### 限制与注意事项

#### 索引池大小限制

```c
// 最小索引池
min_indices = vl * 16 * 2;  // 确保Gather+Scatter各有足够索引
if (index_pool_size < min_indices) index_pool_size = min_indices;
```

#### 最大索引计算

```c
// 使用int64_t元素数量（避免double类型scatter越界）
max_element_idx_64 = buffer_size / sizeof(int64_t) - 1;
max_idx = (max_element_idx_64 < INT32_MAX) ? max_element_idx_64 : INT32_MAX;
```

#### MPI环境配置

必须使用以下参数避免OpenIB/OFI警告：
```bash
--mca btl ^openib --mca mtl ^ofi
```

### 相关文件

| 文件 | 说明 |
|------|------|
| gather_scatter_test.c | 源代码 |
| README_GATHER_SCATTER.md | 详细文档 |
| test_results.md | 测试结果记录 |
| Makefile | 编译配置 |

### 快速理解要点

1. **目的**: 测试SVE非连续内存访问性能
2. **特点**: 汇编内联循环、稀疏度控制、多索引模式
3. **核心**: Gather收集加载 + Scatter分散存储
4. **参数**: 稀疏度最重要（控制访问密度）
5. **MPI**: 支持多进程并行，可选打印所有进程
6. **输出**: 带宽(GB/s) + 时间(ms) + 覆盖率统计

---

## 对话记录要点

### 主要开发历程

1. **初始需求**: 稀疏度参数替代固定索引池
2. **索引模式**: 实现Random/Uniform/Hotspot三种模式
3. **汇编内联**: 循环逻辑完全内置于汇编
4. **Gather+Scatter优化**: 使用相同索引池
5. **MPI增强**: 添加print_all_ranks参数
6. **Uniform修复**: 改进稀疏度<1时的索引分布

### 关键决策

- 稀疏度用double而非整数（支持0.0001-1.0）
- 索引池大小动态计算而非固定
- 汇编使用ARM64分支指令实现循环
- Gather+Scatter语义：dst[idx[i]] = src[idx[i]]
- MPI输出默认仅rank0汇总

### Bug修复

1. **索引越界**: 使用int64_t元素数量计算max_idx
2. **索引池不足**: 设置min_indices = vl * 32
3. **Uniform分布**: 添加remainder分配逻辑

---

## 快速参考卡

```bash
# 编译
gcc -O3 -march=armv9-a+sve -o gather_scatter_test gather_scatter_test.c
mpicc -O3 -march=armv9-a+sve -DUSE_MPI -o gather_scatter_test_mpi gather_scatter_test.c

# 运行
./gather_scatter_test -s <sparsity> -m <mode> -b <buffer_MB>
mpirun --mca btl ^openib --mca mtl ^ofi -np <n> ./gather_scatter_test_mpi [-p]

# 参数
-s: 0.0001-1.0 (稀疏度)
-m: 0(Random)/1(Uniform)/2(Hotspot)
-b: 8-512 (buffer大小MB)
-p: 打印所有MPI进程

# 测试项
0-5: Gather/Scatter/GatherScatter三类
类别名: Gather, Scatter, GatherScatter
```

---

本文档汇总项目核心要点，便于大模型快速理解架构、参数、技术细节和使用方法。