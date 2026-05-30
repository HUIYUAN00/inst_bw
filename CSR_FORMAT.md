# CSR格式详细介绍

## 一、CSR格式基本概念

CSR (Compressed Sparse Row) 是一种稀疏矩阵压缩存储格式，专门用于高效存储和计算稀疏矩阵。

### CSR三数组结构

```
row_ptr[M+1]: 行指针数组
col_idx[NNZ]: 列索引数组  
values[NNZ]: 非零值数组
```

其中：
- M = matrix_size (矩阵维度)
- NNZ = 非零元素数量 (nnz_count)

## 二、代码中的CSR结构定义

### 全局数组声明 (sparse_spmv_test.c:21-27)

```c
static uint64_t *row_ptr = NULL;    // 行指针，长度=M+1
static int32_t *col_idx = NULL;     // 列索引，长度=NNZ
static double *values = NULL;       // 非零值，长度=NNZ
static double *vector = NULL;       // 输入向量，长度=M
static double *result = NULL;       // 输出向量，长度=M
```

### 复数类型扩展 (sparse_spmv_test.c:29-37)

```c
typedef struct {
    double re;  // 实部
    double im;  // 虚部
} complex_double_t;  // 128-bit复数

static complex_double_t *values_complex = NULL;    // 复数非零值
static complex_double_t *vector_complex = NULL;    // 复数向量
static complex_double_t *result_complex = NULL;    // 复数结果
```

## 三、CSR格式构建流程

### Step 1: 生成不重复的2D坐标 (sparse_spmv_test.c:525-555)

```c
sparse_coord_t *coords = malloc(nnz_count * sizeof(sparse_coord_t));
uint64_t *coverage = calloc(...);  // Bitmap去重

// 随机生成阶段
while (unique_count < nnz_count && attempts < max_attempts) {
    uint64_t rand_val = ((uint64_t)rand() << 32) | (uint64_t)rand();
    uint64_t idx = rand_val % max_elements;  // max_elements = M*M
    
    // Bitmap去重检查
    uint64_t bucket = idx / 64;
    uint64_t bit = idx % 64;
    if (!(coverage[bucket] & (1ULL << bit))) {
        coverage[bucket] |= (1ULL << bit);
        coords[unique_count].row = idx / matrix_size;  // 行坐标
        coords[unique_count].col = idx % matrix_size;  // 列坐标
        unique_count++;
    }
}

// 兜底填充阶段（顺序扫描）
for (uint64_t i = 0; i < max_elements && unique_count < nnz_count; i++) {
    if (!(coverage[bucket] & (1ULL << bit))) {
        coords[unique_count].row = i / matrix_size;
        coords[unique_count].col = i % matrix_size;
        unique_count++;
    }
}
```

**关键点：**
- 使用全局Bitmap确保坐标不重复
- 1D索引idx转换为(row, col)：`row = idx / M, col = idx % M`
- 随机生成+顺序兜底确保填满NNZ个位置

### Step 2: 按行优先排序 (sparse_spmv_test.c:559)

```c
qsort(coords, nnz_count, sizeof(sparse_coord_t), compare_coord);

static int compare_coord(const void *a, const void *b) {
    const sparse_coord_t *ca = a;
    const sparse_coord_t *cb = b;
    if (ca->row != cb->row) 
        return (ca->row < cb->row) ? -1 : 1;  // 先按行排序
    return (ca->col < cb->col) ? -1 : (ca->col > cb->col) ? 1 : 0;  // 同行按列排序
}
```

**排序规则：**
- 先按row升序（行优先）
- 同一行内按col升序（列有序）

### Step 3: 构建CSR数组 (sparse_spmv_test.c:561-574)

```c
row_ptr[0] = 0;
uint64_t current_row = 0;

// 填充col_idx并构建row_ptr
for (uint64_t i = 0; i < nnz_count; i++) {
    col_idx[i] = coords[i].col;  // 直接拷贝列索引
    
    // 处理行跳转
    while (current_row < coords[i].row) {
        row_ptr[current_row + 1] = i;  // 空行或行边界
        current_row++;
    }
}

// 处理剩余空行
while (current_row < matrix_size) {
    row_ptr[current_row + 1] = nnz_count;
    current_row++;
}
```

**row_ptr构建逻辑：**
- `row_ptr[0] = 0`：第0行起始位置
- `row_ptr[i+1]`：第i行的结束位置（第i+1行起始）
- 空行：`row_ptr[i] == row_ptr[i+1]`（无非零元素）

## 四、实际示例解析

### 5x5矩阵，40%稀疏度，NNZ=10

**生成的坐标（排序后）：**
```
Index  Row  Col
0      0    2    → row_ptr[1] = 3 (第0行有3个元素)
1      0    3    
2      0    4    
3      1    0    → row_ptr[2] = 5 (第1行有2个元素)
4      1    3    
5      2    2    → row_ptr[3] = 6 (第2行有1个元素)
6      3    1    → row_ptr[4] = 9 (第3行有3个元素)
7      3    3    
8      3    4    
9      4    4    → row_ptr[5] = 10 (第4行有1个元素)
```

**CSR数组：**
```
row_ptr[6] = [0, 3, 5, 6, 9, 10]
             |  |  |  |  |   |
             |  |  |  |  |   最后一个元素位置
             |  |  |  |  第4行结束
             |  |  |  第3行结束
             |  |  第2行结束
             |  第1行结束
             第0行开始

col_idx[10] = [2, 3, 4, 0, 3, 2, 1, 3, 4, 4]
              |--第0行--| |--第1行| |第2| |---第3行---| |4|
              (3个元素)   (2个)    (1)    (3个)       (1)

values[10] = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9]
             与col_idx一一对应
```

### row_ptr语义详解

```
row_ptr[i] → 第i行在col_idx/values中的起始位置
row_ptr[i+1] → 第i行的结束位置（第i+1行起始）

第i行的非零元素：
- 范围: col_idx[row_ptr[i] ... row_ptr[i+1]-1]
- 数量: row_ptr[i+1] - row_ptr[i]

示例：
第0行: col_idx[0..2] = [2, 3, 4], 数量=3-0=3
第1行: col_idx[3..4] = [0, 3],   数量=5-3=2
第2行: col_idx[5]     = [2],     数量=6-5=1
第3行: col_idx[6..8] = [1, 3, 4], 数量=9-6=3
第4行: col_idx[9]     = [4],     数量=10-9=1
```

## 五、SpMV计算过程

### CSR Scalar SpMV算法 (sparse_spmv_test.c:70-82)

```c
for (uint64_t i = 0; i < matrix_size; i++) {
    double sum = 0.0;
    for (uint64_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
        sum += values[j] * vector[col_idx[j]];
    }
    y[i] = sum;
}
```

**计算步骤：**
```
第i行：y[i] = Σ(values[j] * vector[col_idx[j])
       其中j ∈ [row_ptr[i], row_ptr[i+1])

示例（第0行）：
y[0] = values[0]*vector[2] + values[1]*vector[3] + values[2]*vector[4]
     = v0 * vec[2] + v1 * vec[3] + v2 * vec[4]
```

### NEON Complex SpMV算法

**复数乘法：**
```
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i

fcmla #0:  计算ac - bd（实部）
fcmla #90: 计算ad + bc（虚部）
```

**NEON汇编流程：**
```
1. pairs循环（每次处理2个复数）：
   - ldp w5, w6, [x1], #8   → 加载col_idx[2j], col_idx[2j+1]
   - ldr q0, q1, [x2], #16  → 加载values[2j], values[2j+1]（复数）
   - ldr q2, q3             → 加载vector[col_idx[...]]（复数）
   - fcmla v4, v0, v2, #0   → 实部累加
   - fcmla v4, v0, v2, #90  → 虚部累加

2. remainder单元素：
   - ldr w5, [x1], #4       → 加载col_idx[last]
   - ldr q0, [x2], #16      → 加载values[last]
   - ldr q2                 → 加载vector[col_idx[last]]
   - fcmla #0 + #90         → 累加最后一个复数

3. str q4, [y_ptr]          → 存储128-bit复数结果
```

## 六、CSR格式优势

1. **空间效率：**
   - 仅存储非零元素：O(NNZ) vs O(M²)
   - 示例：5x5矩阵，NNZ=10，存储10个值而非25个

2. **计算效率：**
   - 仅对非零元素计算：避免零值乘法
   - SpMV复杂度：O(NNZ) vs O(M²)

3. **行访问友好：**
   - row_ptr直接定位每行范围
   - 适合逐行SpMV计算

4. **SIMD友好：**
   - 同行元素连续存储
   - 易于向量化加载

## 七、关键数据结构总结

```
sparse_coord_t: 临时坐标结构
- row: uint32_t
- col: uint32_t
用于生成和排序阶段，构建完成后释放

CSR三数组：
- row_ptr[M+1]: 行边界指针，构建后不变
- col_idx[NNZ]: 列索引，按行优先排列
- values[NNZ]: 对应非零值（与col_idx同序）

内存布局：
row_ptr: [0, row0_end, row1_end, ..., rowM_end]
         ↑每行起始 = row_ptr[i]
         ↑每行结束 = row_ptr[i+1]
         
col_idx: [row0的列索引 | row1的列索引 | ... | rowM的列索引]
         连续存储，行内有序

values:  [row0的值 | row1的值 | ... | rowM的值]
         与col_idx一一对应
```
