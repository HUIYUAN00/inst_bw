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
 * Hermitian SpMV: y = A*x, A 为 Hermitian 矩阵,CSR 存储全矩阵.
 *
 * 算法结构:
 *   外层循环: 逐行处理 i = 0..dim-1,共 dim 次迭代
 *   内层循环: 向量化处理第 i 行的非零元素 j = row_ptr[i]..row_ptr[i+1]-1
 *
 *   对每个非零元素 A[i][col],通过 col 与 i 的比较判断上/下三角:
 *
 *   上三角 (col >= i) -- 内积操作:
 *     y[i] += a * x[col]
 *     将 a * x[col] 的结果归约累加到 sum_re/sum_im,行结束后写回 y[i]
 *
 *   严格上三角 (col > i) -- 外积操作(逻辑下三角贡献):
 *     y[col] += conj(a) * x[i]
 *     CSR 中存储的是上三角元素 A[i][col],利用 Hermitian 共轭对称性
 *     A[col][i] = conj(A[i][col]),计算的是逻辑下三角 A[col][i] * x[i]
 *     对 y[col] 的贡献,散射累加到 y[col]
 *
 *   下三角 (col < i) -- 跳过:
 *     p1/p2 谓词自动过滤,不产生任何贡献
 *
 * Register allocation:
 *   x6=i  x7=row_start  x8=row_end  x9=j  x10=temp  x11=VL_doubles
 *   x12=unused  x13=unused
 *   q4/v4=vec[i](128bit)
 *   x19=y  x20=val  x21=vec  x22=rp  x23=ci  x24=dim
 *   z0=i-bcast  z1=col_idx(64bit)  z2=val(lo)  z3=val(hi)
 *   z4=x[col](lo)  z5=x[i]-bcast  z6=x[col](hi)  z7=result(hi)
 *   z8=col*2  z9=col*2+1  z10=temp  z11=内积高半累加器
 *   z14=temp  z15=内积低半累加器
 *   p0=loop/reduce-p_re  p1=col>=i/reduce-p_im  p2=col>i
 *   p3=zip1(p1,p1)  p4=zip1(p2,p2)  p5=zip2(p1,p1)  p6=zip2(p2,p2)
 *   p7=temp  p8=odd-lanes(循环不变量,归约分离实虚部)
 */

spmv_standard:
	/* 函数序言:保存被调用者保存的寄存器 */
	stp x29, x30, [sp, #-16]!   /* 保存帧指针和返回地址 */
	stp x19, x20, [sp, #-16]!   /* 保存 x19-x20,用于存储参数指针 */
	stp x21, x22, [sp, #-16]!   /* 保存 x21-x22 */
	stp x23, x24, [sp, #-16]!   /* 保存 x23-x24 */
	mov x29, sp                  /* 设置帧指针 */

	/* 保存函数参数到被调用者保存的寄存器,避免在内层循环中被覆盖 */
	mov x19, x0                  /* x19 = result_ptr (y) */
	mov x20, x1                  /* x20 = values_ptr (val) */
	mov x21, x2                  /* x21 = vector_ptr (vec) */
	mov x22, x3                  /* x22 = row_ptr (rp) */
	mov x23, x4                  /* x23 = col_idx (ci) */
	mov x24, x5                  /* x24 = matrix_dim (dim) */

	/* ===== Phase 1: 清零结果向量 y ===== */
	/* 操作数据: y[0..2*dim) 个 double,即 dim 个复数 */
	/* 目的: 初始化 y 为零向量,为后续累加做准备 */
	mov x6, #0                   /* x6 = 偏移量,初始为 0 */
	lsl x7, x24, #1              /* x7 = 2*dim,总共需要清零的 double 数量 */
	rdvl x11, #1                 /* x11 = SVE 向量长度(字节数) */
	lsr x11, x11, #3             /* x11 = VL_doubles,每次迭代处理的 double 数量 */
1:
	whilelt p1.d, x6, x7         /* p1 = 谓词,标记 x6 < x7 的有效 lane */
	beq 2f                       /* 如果所有 lane 都无效(x6 >= x7),退出循环 */
	mov z1.d, #0                 /* z1 = 全零向量 */
	st1d z1.d, p1, [x19, x6, lsl #3]  /* 将零存储到 y[x6..x6+VL),lsl #3 = *8 字节 */
	add x6, x6, x11              /* x6 += VL_doubles,前进到下一批 */
	b 1b                         /* 继续循环 */
2:

	/* ===== Phase 2: 逐行处理外层循环 ===== */
	/* 遍历每一行 i = 0..dim-1,共 dim 次迭代 */
	/* 每次迭代处理第 i 行的所有非零元素,通过 col 与 i 的比较区分上/下三角 */
	/* 计算循环不变量: odd-lane 谓词,用于共轭乘法符号修正 */
	ptrue p7.b                     /* p7 = 全真谓词 */
	index z0.d, #0, #1             /* z0 = [0, 1, 2, 3, ...] */
	and z0.d, z0.d, #1             /* z0 = [0, 1, 0, 1, ...] */
	cmpne p8.d, p7/z, z0.d, #0     /* p8 = odd lanes (虚部位置,循环不变量) */
	mov x6, #0                   /* x6 = i,当前行号 */
3:
	cmp x6, x24                  /* 比较 i 与 dim */
	bge 4f                       /* 如果 i >= dim,退出外层循环 */

	/* 加载当前行的 CSR 指针范围 */
	/* 操作数据: row_ptr[i] 和 row_ptr[i+1] */
	add x10, x22, x6, lsl #3     /* x10 = &row_ptr[i],lsl #3 = *8 字节 */
	ldr x7, [x10]                /* x7 = row_ptr[i],该行非零元素的起始索引 */
	ldr x8, [x10, #8]            /* x8 = row_ptr[i+1],该行非零元素的结束索引 */

	/* 加载当前行的输入向量元素 vec[i] */
	/* 操作数据: vec[i] = [re, im],128 位整体加载到 q4 */
	add x10, x21, x6, lsl #4     /* x10 = &vec[i],lsl #4 = *16 字节(complex_double_t) */
	ldr q4, [x10]                /* q4 = vec[i] = [re, im],128 位整体加载 */
	mov z5.q, q4                 /* z5 = x[i] interleaved [re, im, re, im, ...], 持久广播 */

	/* 初始化累加器 */
	mov z15.d, #0                /* z15 = 内积低半累加器 [re, im, ...], 初始为 0 */
	mov z11.d, #0                /* z11 = 内积高半累加器 [re, im, ...], 初始为 0 */

	/* ----- 内层循环: 向量化处理第 i 行的非零元素 j = row_start..row_end-1 ----- */
	/* 每次迭代处理 VL_doubles 个非零元素 */
	/* 对每个元素 A[i][col]: */
	/*   col >= i (上三角): 内积操作 y[i] += a * x[col] */
	/*   col >  i (严格上三角): 外积操作(逻辑下三角贡献)y[col] += conj(a) * x[i] */
	/*   col <  i (下三角): 跳过 */
	mov x9, x7                   /* x9 = j = row_ptr[i],当前处理的非零元素索引 */
5:
	cmp x9, x8                   /* 比较 j 与 row_end */
	bge 6f                       /* 如果 j >= row_end,退出内层循环 */

	/* 加载列号并创建谓词 */
	add x10, x23, x9, lsl #2     /* x10 = &col_idx[j], x9*4 bytes offset */
	whilelt p0.d, x9, x8         /* p0 = 谓词,标记 j < row_end 的有效 lane */
	ld1sw z1.d, p0/z, [x10]      /* z1 = col_idx[j..j+VL), sequential load */
	                              /* ld1sw: load int32 and sign-extend to int64 */
	add x10, x20, x9, lsl #4     /* x10 = &val[j], x9*16 bytes offset */

	/* 创建上三角判断谓词 */
	dup z0.d, x6                 /* z0 = [i, i, i, ...],广播当前行号 */
	cmpge p1.d, p0/z, z1.d, z0.d /* p1 = col >= i,上三角(含对角线)谓词 */
	cmpgt p2.d, p0/z, z1.d, z0.d /* p2 = col > i,严格上三角谓词 */

	/* 扩展谓词用于复数运算 */
	/* 每个复数占用 2 个 double lane (re, im),需要将谓词从"复数粒度"扩展到"double 粒度" */
	zip1 p3.d, p1.d, p1.d        /* p3 = zip1(p1, p1),低半部分的 col>=i 谓词 */
	zip2 p5.d, p1.d, p1.d        /* p5 = zip2(p1, p1),高半部分的 col>=i 谓词 */
	zip1 p4.d, p2.d, p2.d        /* p4 = zip1(p2, p2),低半部分的 col>i 谓词 */
	zip2 p6.d, p2.d, p2.d        /* p6 = zip2(p2, p2),高半部分的 col>i 谓词 */

	/* 顺序加载矩阵值 val[j],内存中本身按 [re, im, re, im, ...] 顺序存储 */
	ld1d z2.d, p3/z, [x10]       /* z2 = val[j..j+VL/2), 顺序读取前半部分 */
	add x10, x10, x11, lsl #3    /* x10 = &val[j+VL_doubles/2], 前进到高半部分基址 */
	ld1d z3.d, p5/z, [x10]       /* z3 = val[j+VL/2..j+VL), 顺序读取后半部分 */

	/* 计算输入向量的偏移并 gather 加载 vec[col] */
	lsl z8.d, z1.d, #1           /* z8 = col*2, vec[col] 的 double 索引 */
	add x10, x21, #8             /* x10 = &vec[0].im, 虚部基址 */
	ld1d z4.d, p1/z, [x21, z8.d, lsl #3]  /* z4 = vec[col].re, gather load (p1: col>=i) */
	ld1d z6.d, p1/z, [x10, z8.d, lsl #3]  /* z6 = vec[col].im, gather load (p1: col>=i) */
	zip1 z10.d, z4.d, z6.d       /* z10 = x[col] low, interleaved [re, im, ...] */
	zip2 z4.d, z4.d, z6.d        /* z4 = x[col] high, interleaved [re, im, ...] */

	/* ===== 内积: a * x[col], fcmla 复数乘法, 仅 col >= i (p3/p5 掩码) ===== */
	/* fcmla #0:  Zd.even += Zn1.even * Zn2.even (re*re) */
	/*            Zd.odd  += Zn1.odd  * Zn2.even (im*re) */
	/* fcmla #90: Zd.even -= Zn1.odd  * Zn2.odd  (-im*im) */
	/*            Zd.odd  += Zn1.even * Zn2.odd  (re*im) */
	/* 结果: even = re*re - im*im, odd = im*re + re*im */
	/* 直接累加到 z15(低半)/z11(高半),p3/p5 互斥,无需每次清零 */
	fcmla z15.d, p3/m, z2.d, z10.d, #0   /* z15 += val * x[col] (re*re, im*re) */
	fcmla z15.d, p3/m, z2.d, z10.d, #90  /* z15 += rotated(-im, re) * x[col] */
	fcmla z11.d, p5/m, z3.d, z4.d, #0    /* z11 += val * x[col] (re*re, im*re) */
	fcmla z11.d, p5/m, z3.d, z4.d, #90   /* z11 += rotated(-im, re) * x[col] */

	/* ===== 外积散射(逻辑下三角贡献): y[col] += conj(a) * x[i], 仅 col > i (p4/p6 掩码) ===== */
	/* fcmla #0:   Zd.even += Zn1.even * Zn2.even (re*re) */
	/*             Zd.odd  += Zn1.odd  * Zn2.even (im*re) */
	/* fcmla #270: Zd.even += Zn1.odd  * Zn2.odd  (im*im) */
	/*             Zd.odd  -= Zn1.even * Zn2.odd  (-re*im) */
	/* 结果: even = re*re + im*im, odd = im*re - re*im = conj(a)*x */
	/* 先加载 y[col] 到 z6/z7,fcmla 直接在其上累加,省去 fadd */
	mov z9.d, z8.d               /* z9 = col*2 */
	add z9.d, z9.d, #1           /* z9 = col*2+1, 散射交错索引 */
	/* 低半部分 */
	zip1 z10.d, z8.d, z9.d       /* z10 = [col0*2, col0*2+1, col1*2, col1*2+1, ...] */
	ld1d z6.d, p4/z, [x19, z10.d, lsl #3]  /* z6 = y[col] 当前值 */
	fcmla z6.d, p4/m, z2.d, z5.d, #0    /* z6 += val * x[i] (re*re, im*re) */
	fcmla z6.d, p4/m, z2.d, z5.d, #270  /* z6 += conj correction (im*im, -re*im) */
	st1d z6.d, p4, [x19, z10.d, lsl #3]  /* 存储回 y[col] */
	/* 高半部分 */
	zip2 z10.d, z8.d, z9.d       /* z10 = 高半部分的索引 */
	ld1d z7.d, p6/z, [x19, z10.d, lsl #3]  /* z7 = y[col] 当前值 */
	fcmla z7.d, p6/m, z3.d, z5.d, #0    /* z7 += val * x[i] (re*re, im*re) */
	fcmla z7.d, p6/m, z3.d, z5.d, #270  /* z7 += conj correction (im*im, -re*im) */
	st1d z7.d, p6, [x19, z10.d, lsl #3]  /* 存储回 y[col] */

	/* 内层循环迭代 */
	add x9, x9, x11              /* j += VL_doubles,前进到下一批非零元素 */
	b 5b                         /* 继续内层循环 */
6:

	/* 行结束: 归约 z15(低半)/z11(高半) 累加器,写回 y[i] */
	/* z15/z11 非活跃 lane 保持为 0,无需谓词掩码 */
	ptrue p7.b                   /* p7 = 全真谓词 */
	not p0.b, p7/z, p8.b         /* p0 = even lanes (实部位置) */
	faddv d0, p0, z15.d          /* d0 = sum of z15 even lanes (低半 re) */
	faddv d1, p0, z11.d          /* d1 = sum of z11 even lanes (高半 re) */
	fadd d0, d0, d1              /* d0 = total sum_re */
	add x10, x19, x6, lsl #4     /* x10 = &y[i],lsl #4 = *16 字节 */
	ldr d4, [x10]                /* d4 = y[i].re 当前值 */
	fadd d4, d4, d0              /* d4 += sum_re */
	str d4, [x10]                /* 存储 y[i].re */
	mov p7.b, p8.b               /* p7 = odd lanes (faddv 仅支持 p0-p7) */
	faddv d0, p7, z15.d          /* d0 = sum of z15 odd lanes (低半 im) */
	faddv d1, p7, z11.d          /* d1 = sum of z11 odd lanes (高半 im) */
	fadd d0, d0, d1              /* d0 = total sum_im */
	ldr d4, [x10, #8]            /* d4 = y[i].im 当前值 */
	fadd d4, d4, d0              /* d4 += sum_im */
	str d4, [x10, #8]            /* 存储 y[i].im */

	/* 外层循环迭代 */
	add x6, x6, #1               /* i++,前进到下一行 */
	b 3b                         /* 继续外层循环 */
4:

	/* 函数尾声:恢复被调用者保存的寄存器 */
	ldp x23, x24, [sp], #16      /* 恢复 x23-x24 */
	ldp x21, x22, [sp], #16      /* 恢复 x21-x22 */
	ldp x19, x20, [sp], #16      /* 恢复 x19-x20 */
	ldp x29, x30, [sp], #16      /* 恢复帧指针和返回地址 */
	ret                          /* 返回调用者 */

	.size spmv_standard, .-spmv_standard
