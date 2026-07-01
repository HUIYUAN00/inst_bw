	.arch armv8.3-a+sve
	.text
	.global spmv_standard
	.type spmv_standard, %function

/*
 * void spmv_standard(
 *     void *result_ptr,    // x0: y (complex_double_t*, 2*dim doubles)
 *     void *values_ptr,    // x1: val (complex_double_t*, nnz entries)
 *     void *vector_ptr,    // x2: vec (complex_double_t*, dim entries)
 *     uint64_t *row_ptr,   // x3: CSR row pointers (dim+1 entries)
 *     int32_t *col_idx,    // x4: CSR column indices (nnz entries)
 *     int matrix_dim       // x5: matrix dimension
 * )
 *
 * Hermitian SpMV: y = A*x, A 为 Hermitian 矩阵，CSR 存储全矩阵。
 *
 * 算法结构:
 *   外层循环: 逐行处理 i = 0..dim-1，共 dim 次迭代
 *   内层循环: 向量化处理第 i 行的非零元素 j = row_ptr[i]..row_ptr[i+1]-1
 *
 *   对每个非零元素 A[i][col]，通过 col 与 i 的比较判断上/下三角:
 *
 *   上三角 (col >= i) — 内积操作:
 *     y[i] += a * x[col]
 *     将 a * x[col] 的结果归约累加到 sum_re/sum_im，行结束后写回 y[i]
 *
 *   严格上三角 (col > i) — 外积操作:
 *     y[col] += conj(a) * x[i]
 *     利用 Hermitian 共轭对称性，将 conj(a) * x[i] 散射累加到 y[col]
 *
 *   下三角 (col < i) — 跳过:
 *     p1/p2 谓词自动过滤，不产生任何贡献
 *
 * Register allocation:
 *   x6=i  x7=row_start  x8=row_end  x9=j  x10=temp  x11=VL_doubles
 *   x12=sum_re  x13=sum_im  x14=xi_re(bits)  x15=xi_im(bits)
 *   x19=y  x20=val  x21=vec  x22=rp  x23=ci  x24=dim
 *   z0=index/i-bcast  z1=col_idx(64bit)  z2=a_re  z3=a_im
 *   z4=xj_re  z5=xj_im  z6=result(lo)  z7=result(hi)
 *   z8=col*2  z9=col*2+1  z10/z11=temp  z12=xi_re-bcast  z13=xi_im-bcast
 *   z14=y[col]  z15=unused
 *   p0=loop/reduce-p_re  p1=col>=i/reduce-p_im  p2=col>i
 *   p3=zip1(p1,p1)  p4=zip1(p2,p2)  p5=zip2(p1,p1)  p6=zip2(p2,p2)
 *   p7=temp
 */

spmv_standard:
	/* 函数序言：保存被调用者保存的寄存器 */
	stp x29, x30, [sp, #-16]!   /* 保存帧指针和返回地址 */
	stp x19, x20, [sp, #-16]!   /* 保存 x19-x20，用于存储参数指针 */
	stp x21, x22, [sp, #-16]!   /* 保存 x21-x22 */
	stp x23, x24, [sp, #-16]!   /* 保存 x23-x24 */
	mov x29, sp                  /* 设置帧指针 */

	/* 保存函数参数到被调用者保存的寄存器，避免在内层循环中被覆盖 */
	mov x19, x0                  /* x19 = result_ptr (y) */
	mov x20, x1                  /* x20 = values_ptr (val) */
	mov x21, x2                  /* x21 = vector_ptr (vec) */
	mov x22, x3                  /* x22 = row_ptr (rp) */
	mov x23, x4                  /* x23 = col_idx (ci) */
	mov x24, x5                  /* x24 = matrix_dim (dim) */

	/* ===== Phase 1: 清零结果向量 y ===== */
	/* 操作数据: y[0..2*dim) 个 double，即 dim 个复数 */
	/* 目的: 初始化 y 为零向量，为后续累加做准备 */
	mov x6, #0                   /* x6 = 偏移量，初始为 0 */
	lsl x7, x24, #1              /* x7 = 2*dim，总共需要清零的 double 数量 */
	rdvl x11, #1                 /* x11 = SVE 向量长度（字节数） */
	lsr x11, x11, #3             /* x11 = VL_doubles，每次迭代处理的 double 数量 */
1:
	whilelt p1.d, x6, x7         /* p1 = 谓词，标记 x6 < x7 的有效 lane */
	beq 2f                       /* 如果所有 lane 都无效（x6 >= x7），退出循环 */
	mov z1.d, #0                 /* z1 = 全零向量 */
	st1d z1.d, p1, [x19, x6, lsl #3]  /* 将零存储到 y[x6..x6+VL)，lsl #3 = *8 字节 */
	add x6, x6, x11              /* x6 += VL_doubles，前进到下一批 */
	b 1b                         /* 继续循环 */
2:

	/* ===== Phase 2: 逐行处理外层循环 ===== */
	/* 遍历每一行 i = 0..dim-1，共 dim 次迭代 */
	/* 每次迭代处理第 i 行的所有非零元素，通过 col 与 i 的比较区分上/下三角 */
	mov x6, #0                   /* x6 = i，当前行号 */
3:
	cmp x6, x24                  /* 比较 i 与 dim */
	bge 4f                       /* 如果 i >= dim，退出外层循环 */

	/* 加载当前行的 CSR 指针范围 */
	/* 操作数据: row_ptr[i] 和 row_ptr[i+1] */
	add x10, x22, x6, lsl #3     /* x10 = &row_ptr[i]，lsl #3 = *8 字节 */
	ldr x7, [x10]                /* x7 = row_ptr[i]，该行非零元素的起始索引 */
	ldr x8, [x10, #8]            /* x8 = row_ptr[i+1]，该行非零元素的结束索引 */

	/* 加载当前行的输入向量元素 vec[i] */
	/* 操作数据: vec[i].re 和 vec[i].im */
	add x10, x21, x6, lsl #4     /* x10 = &vec[i]，lsl #4 = *16 字节（complex_double_t） */
	ldr d0, [x10]                /* d0 = vec[i].re */
	ldr d1, [x10, #8]            /* d1 = vec[i].im */
	fmov x14, d0                 /* x14 = vec[i].re 的位模式，保存到 GPR 供后续广播 */
	fmov x15, d1                 /* x15 = vec[i].im 的位模式，保存到 GPR 供后续广播 */

	/* 初始化累加器 */
	mov x12, #0                  /* x12 = sum_re = 0，累加 y[i].re 的贡献 */
	mov x13, #0                  /* x13 = sum_im = 0，累加 y[i].im 的贡献 */

	/* ----- 内层循环: 向量化处理第 i 行的非零元素 j = row_start..row_end-1 ----- */
	/* 每次迭代处理 VL_doubles 个非零元素 */
	/* 对每个元素 A[i][col]: */
	/*   col >= i (上三角): 内积操作 y[i] += a * x[col] */
	/*   col >  i (严格上三角): 外积操作 y[col] += conj(a) * x[i] */
	/*   col <  i (下三角): 跳过 */
	mov x9, x7                   /* x9 = j = row_ptr[i]，当前处理的非零元素索引 */
5:
	cmp x9, x8                   /* 比较 j 与 row_end */
	bge 6f                       /* 如果 j >= row_end，退出内层循环 */

	/* 创建索引向量并收集列号 */
	/* 操作数据: col_idx[j..j+VL) */
	index z0.d, x9, #1           /* z0 = [j, j+1, j+2, ...]，创建索引序列 */
	lsl z10.d, z0.d, #1          /* z10 = [2j, 2(j+1), ...]，val.re 的偏移（*2 因为复数） */
	mov z11.d, z10.d             /* z11 = z10 */
	add z11.d, z11.d, #1         /* z11 = [2j+1, 2(j+1)+1, ...]，val.im 的偏移 */
	whilelt p0.d, x9, x8         /* p0 = 谓词，标记 j < row_end 的有效 lane */
	ld1sw z1.d, p0/z, [x23, z0.d, lsl #2]  /* z1 = col_idx[j..j+VL) */
	                                      /* ld1sw: 加载 int32 并符号扩展为 int64 */
	                                      /* lsl #2 = *4 字节（int32_t 大小） */

	/* 创建上三角判断谓词 */
	/* 操作数据: col_idx 和当前行号 i */
	/* 目的: 判断每个非零元素是否在上三角（col >= i）或严格上三角（col > i） */
	dup z0.d, x6                 /* z0 = [i, i, i, ...]，广播当前行号 */
	cmpge p1.d, p0/z, z1.d, z0.d /* p1 = col >= i，上三角（含对角线）谓词 */
	cmpgt p2.d, p0/z, z1.d, z0.d /* p2 = col > i，严格上三角谓词 */

	/* 扩展谓词用于复数运算 */
	/* 目的: 每个复数占用 2 个 double lane (re, im)，需要将谓词从"复数粒度"扩展到"double 粒度" */
	zip1 p3.d, p1.d, p1.d        /* p3 = zip1(p1, p1)，低半部分的 col>=i 谓词 */
	zip2 p5.d, p1.d, p1.d        /* p5 = zip2(p1, p1)，高半部分的 col>=i 谓词 */
	zip1 p4.d, p2.d, p2.d        /* p4 = zip1(p2, p2)，低半部分的 col>i 谓词 */
	zip2 p6.d, p2.d, p2.d        /* p6 = zip2(p2, p2)，高半部分的 col>i 谓词 */

	/* 收集矩阵值 val[j] */
	/* 操作数据: val[j].re 和 val[j].im，使用 p1 掩码仅收集上三角元素 */
	ld1d z2.d, p1/z, [x20, z10.d, lsl #3]  /* z2 = val[j].re，gather 加载 */
	ld1d z3.d, p1/z, [x20, z11.d, lsl #3]  /* z3 = val[j].im，gather 加载 */
	                                        /* lsl #3 = *8 字节（double 大小） */

	/* 计算输入向量的偏移 */
	/* 操作数据: col_idx，计算 vec[col] 的索引 */
	lsl z8.d, z1.d, #1             /* z8 = col*2，vec[col].re 的索引 */
	mov z9.d, z8.d                 /* z9 = z8 */
	add z9.d, z9.d, #1             /* z9 = col*2+1，vec[col].im 的索引 */

	/* 收集输入向量 vec[col] */
	/* 操作数据: vec[col].re 和 vec[col].im，使用 p1 掩码仅收集上三角对应的元素 */
	ld1d z4.d, p1/z, [x21, z8.d, lsl #3]  /* z4 = vec[col].re，gather 加载 */
	ld1d z5.d, p1/z, [x21, z9.d, lsl #3]  /* z5 = vec[col].im，gather 加载 */

	/* ===== 内积操作: prod = a * x[col]，仅 col >= i 的元素参与 (p1 掩码) ===== */
	/* 操作数据: z2=a_re, z3=a_im, z4=xj_re, z5=xj_im */
	/* 计算公式: */
	/*   prod_re = a_re * xj_re - a_im * xj_im */
	/*   prod_im = a_re * xj_im + a_im * xj_re */
	/* 结果归约累加到 sum_re/sum_im，行结束后写回 y[i] */
	fmul z6.d, z2.d, z4.d          /* z6 = a_re * xj_re */
	fmls z6.d, p1/m, z3.d, z5.d    /* z6 -= a_im * xj_im，p1 掩码保护 */
	fmul z7.d, z2.d, z5.d          /* z7 = a_re * xj_im */
	fmla z7.d, p1/m, z3.d, z4.d    /* z7 += a_im * xj_re，p1 掩码保护 */

	/* 交错实部和虚部为 [re, im, re, im, ...] 格式 */
	/* 目的: 便于后续归约和存储 */
	zip1 z10.d, z6.d, z7.d         /* z10 = 低半部分交错 [re0, im0, re1, im1, ...] */
	zip2 z7.d, z6.d, z7.d          /* z7 = 高半部分交错 */
	mov z6.d, z10.d                /* z6 = 低半部分结果 */

	/* 内积归约: 将向量结果累加到标量 sum_re (x12) 和 sum_im (x13) */
	/* 操作数据: z6 (低半交错结果), z7 (高半交错结果) */
	/* 目的: 将当前批次所有 col >= i 的 a * x[col] 贡献累加，行结束后写回 y[i] */
	
	/* 创建 even/odd lane 谓词用于分离实部和虚部 */
	ptrue p7.b                     /* p7 = 全真谓词 */
	index z0.d, #0, #1             /* z0 = [0, 1, 2, 3, ...] */
	and z0.d, z0.d, #1             /* z0 = [0, 1, 0, 1, ...] */
	cmpne p1.d, p7/z, z0.d, #0     /* p1 = odd lanes (虚部位置) */
	not p0.b, p7/z, p1.b           /* p0 = even lanes (实部位置) */

	/* 低半部分归约 */
	and p7.b, p3/z, p3.b, p0.b     /* p7 = p3 AND p0，有效的实部 lane */
	faddv d0, p7, z6.d             /* d0 = sum of z6[even lanes] */
	fmov d1, x12                   /* d1 = 当前 sum_re */
	fadd d0, d0, d1                /* d0 += sum_re */
	fmov x12, d0                   /* x12 = 新的 sum_re */
	and p7.b, p3/z, p3.b, p1.b     /* p7 = p3 AND p1，有效的虚部 lane */
	faddv d0, p7, z6.d             /* d0 = sum of z6[odd lanes] */
	fmov d1, x13                   /* d1 = 当前 sum_im */
	fadd d0, d0, d1                /* d0 += sum_im */
	fmov x13, d0                   /* x13 = 新的 sum_im */

	/* 高半部分归约 */
	and p7.b, p5/z, p5.b, p0.b     /* p7 = p5 AND p0，有效的实部 lane */
	faddv d0, p7, z7.d             /* d0 = sum of z7[even lanes] */
	fmov d1, x12                   /* d1 = 当前 sum_re */
	fadd d0, d0, d1                /* d0 += sum_re */
	fmov x12, d0                   /* x12 = 新的 sum_re */
	and p7.b, p5/z, p5.b, p1.b     /* p7 = p5 AND p1，有效的虚部 lane */
	faddv d0, p7, z7.d             /* d0 = sum of z7[odd lanes] */
	fmov d1, x13                   /* d1 = 当前 sum_im */
	fadd d0, d0, d1                /* d0 += sum_im */
	fmov x13, d0                   /* x13 = 新的 sum_im */

	/* ===== 外积操作: conj(a) * x[i]，仅 col > i 的元素参与 (p2 掩码) ===== */
	/* 操作数据: z2=a_re, z3=a_im, x14=xi_re, x15=xi_im */
	/* 目的: 利用 Hermitian 共轭对称性 A[col][i] = conj(A[i][col])， */
	/*       将 conj(a) * x[i] 散射累加到 y[col] */
	/* 计算公式: */
	/*   conj_prod_re = a_re * xi_re + a_im * xi_im */
	/*   conj_prod_im = a_re * xi_im - a_im * xi_re */
	dup z12.d, x14                 /* z12 = [xi_re, xi_re, ...]，广播 x[i].re */
	dup z13.d, x15                 /* z13 = [xi_im, xi_im, ...]，广播 x[i].im */

	fmul z6.d, z2.d, z12.d         /* z6 = a_re * xi_re */
	fmla z6.d, p2/m, z3.d, z13.d   /* z6 += a_im * xi_im，p2 掩码（仅 col > i） */
	fmul z10.d, z2.d, z13.d        /* z10 = a_re * xi_im */
	fmls z10.d, p2/m, z3.d, z12.d  /* z10 -= a_im * xi_re，p2 掩码 */

	/* 交错实部和虚部 */
	zip1 z11.d, z6.d, z10.d        /* z11 = 低半部分交错 [re, im, re, im, ...] */
	zip2 z7.d, z6.d, z10.d         /* z7 = 高半部分交错 */
	mov z6.d, z11.d                /* z6 = 低半部分结果 */

	/* 外积散射存储: y[col] += conj(a) * x[i]，仅 col > i (p4/p6 掩码) */
	/* 操作数据: y[col]，gather 加载 → fadd → scatter 存储 */
	
	/* 低半部分散射 */
	zip1 z10.d, z8.d, z9.d         /* z10 = [col0*2, col0*2+1, col1*2, col1*2+1, ...] */
	ld1d z14.d, p4/z, [x19, z10.d, lsl #3]  /* z14 = y[col]，gather 加载当前值 */
	fadd z14.d, p4/m, z14.d, z6.d  /* z14 += z6（共轭乘法的低半结果） */
	st1d z14.d, p4, [x19, z10.d, lsl #3]    /* 存储回 y[col] */

	/* 高半部分散射 */
	zip2 z10.d, z8.d, z9.d         /* z10 = 高半部分的索引 */
	ld1d z14.d, p6/z, [x19, z10.d, lsl #3]  /* z14 = y[col]，gather 加载 */
	fadd z14.d, p6/m, z14.d, z7.d  /* z14 += z7（共轭乘法的高半结果） */
	st1d z14.d, p6, [x19, z10.d, lsl #3]    /* 存储回 y[col] */

	/* 内层循环迭代 */
	add x9, x9, x11              /* j += VL_doubles，前进到下一批非零元素 */
	b 5b                         /* 继续内层循环 */
6:

	/* 行结束: 将内积累加的 sum_re/sum_im 写回 y[i] */
	/* 操作数据: y[i].re 和 y[i].im */
	/* 目的: y[i] += 当前行所有 col >= i 的 a * x[col] 贡献之和 */
	add x10, x19, x6, lsl #4     /* x10 = &y[i]，lsl #4 = *16 字节 */
	ldr d4, [x10]                /* d4 = y[i].re 当前值 */
	fmov d5, x12                 /* d5 = sum_re */
	fadd d4, d4, d5              /* d4 += sum_re */
	str d4, [x10]                /* 存储 y[i].re */
	ldr d4, [x10, #8]            /* d4 = y[i].im 当前值 */
	fmov d5, x13                 /* d5 = sum_im */
	fadd d4, d4, d5              /* d4 += sum_im */
	str d4, [x10, #8]            /* 存储 y[i].im */

	/* 外层循环迭代 */
	add x6, x6, #1               /* i++，前进到下一行 */
	b 3b                         /* 继续外层循环 */
4:

	/* 函数尾声：恢复被调用者保存的寄存器 */
	ldp x23, x24, [sp], #16      /* 恢复 x23-x24 */
	ldp x21, x22, [sp], #16      /* 恢复 x21-x22 */
	ldp x19, x20, [sp], #16      /* 恢复 x19-x20 */
	ldp x29, x30, [sp], #16      /* 恢复帧指针和返回地址 */
	ret                          /* 返回调用者 */

	.size spmv_standard, .-spmv_standard
