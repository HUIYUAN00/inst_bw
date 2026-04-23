# SVE Gather/Scatter 内存带宽测试工具

基于 ARM SVE (Scalable Vector Extension) 的 Gather/Scatter 指令性能测试工具。

## 功能特性

- **Gather 测试**：测试 SVE 向量收集加载指令 (LD1W/LD1SW/LD1D)
- **Scatter 测试**：测试 SVE 向量分散存储指令 (ST1W/ST1D)
- **Gather+Scatter 组合测试**：测试完全非连续内存操作
- **参数可配置**：缓冲区大小、下标池大小、迭代次数均可通过命令行控制
- **结果验证**：内置结果验证机制，确保测试准确性

## 编译

```bash
make gather_scatter_test
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
| `-i, --index-size <K>` | 下标池大小 (K 个元素) | 1024 |
| `-w, --warmup <N>` | 预热迭代次数 | 5 |
| `-t, --test <N>` | 测试迭代次数 | 10 |

### 测试选择

| 方式 | 说明 |
|------|------|
| `<index>` | 按索引号选择测试 (0-5) |
| `<name>` | 按名称部分匹配 |
| `<category>` | 按类别选择 (Gather/Scatter/GatherScatter) |

## 测试项说明

| 索引 | 测试名称 | 类别 | 说明 |
|------|----------|------|------|
| 0 | SVE Gather LD1W | Gather | 使用 LD1W 指令按索引从随机地址加载 32 位数据，顺序存储 |
| 1 | SVE Gather LD1SW+LD1D | Gather | 使用 LD1SW 加载有符号 32 位扩展到 64 位，配合 LD1D 收集数据 |
| 2 | SVE Scatter ST1W | Scatter | 顺序加载 32 位数据，使用 ST1W 按索引分散存储到随机地址 |
| 3 | SVE Scatter ST1D | Scatter | 顺序加载 64 位数据，使用 ST1D 分散存储，配合 LD1SW 加载索引 |
| 4 | SVE Gather+Scatter W | GatherScatter | 完全非连续操作：使用 LD1W 收集 + ST1W 分散 (32 位) |
| 5 | SVE Gather+Scatter D | GatherScatter | 完全非连续操作：使用 LD1D 收集 + ST1D 分散 (64 位) |

## 使用示例

```bash
# 默认参数运行所有测试
./gather_scatter_test

# 自定义缓冲区和下标池大小
./gather_scatter_test -b 64 -i 512

# 小缓冲区快速测试
./gather_scatter_test -b 16 -i 128 -w 1 -t 3

# 大缓冲区测试 Gather 类别
./gather_scatter_test -b 512 -i 2048 Gather

# 按索引运行特定测试
./gather_scatter_test 0 2 4

# 按类别运行 Scatter 测试
./gather_scatter_test Scatter

# 高精度测试 (多次迭代)
./gather_scatter_test -w 10 -t 50
```

## 输出说明

``============================================================
SVE Gather/Scatter Bandwidth Benchmark
============================================================
SVE Vector Length: 32 bytes (256 bits)
Buffer Size: 128 MB per array
Index Pool Size: 1048576 elements
Warmup Iterations: 5
Test Iterations: 10
Registered Tests: 6

Test                          Category       GB/s   Time(ms)   Data(MB)
============================================================
SVE Gather LD1W                 Gather       3.60     74.497        256
```

- **Test**: 测试项名称
- **Category**: 测试类别
- **GB/s**: 带宽 (吉字节/秒)
- **Time(ms)**: 单次测试执行时间 (毫秒)
- **Data(MB)**: 单次测试处理的数据量 (兆字节)

## 参数选择建议

### 缓冲区大小 (-b)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 快速测试 | 16-32 | 快速验证功能 |
| 标准测试 | 128-256 | 平衡测试时间和准确度 |
| 大内存测试 | 512-1024 | 测试大容量内存性能 |

### 下标池大小 (-i)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 小范围随机访问 | 128-256 | 模拟紧凑数据结构访问 |
| 中等范围 | 1024 | 平衡覆盖范围和内存占用 |
| 大范围随机访问 | 2048+ | 测试更分散的访问模式 |

**注意**: 下标池大小应小于缓冲区元素数，否则会产生重复索引。

### 迭代次数 (-w/-t)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 快速验证 | -w 1 -t 3 | 快速功能验证 |
| 标准测试 | -w 5 -t 10 | 默认设置 |
| 高精度 | -w 10 -t 50 | 减少测量波动 |

## 技术细节

### 下标生成

下标池使用随机步长生成，确保均匀覆盖整个缓冲区范围：

```c
stride = buffer_elements / index_pool_size
index[i] = i * stride + rand() % stride
```

### ld1sw 指令优化

在 64 位操作中使用 `ld1sw` 替代 `ld1w + sunpklo`：
- 直接加载 32 位有符号整数并扩展为 64 位
- 减少指令数量，提高效率

### 结果验证

所有 Gather/Scatter 测试包含结果验证：
- Gather: 验证收集的数据是否与源数据匹配
- Scatter: 统计每个位置的写入次数，验证正确性
- Gather+Scatter: 双向验证

验证失败时会输出 `VERIFY_FAIL(n)` 标记。

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