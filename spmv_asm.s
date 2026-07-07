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
 *     int matrix_dim,      // x5: matrix dimension
 *     complex_double_t alpha  // d0/d1: alpha (re, im), HFA in FP registers
 * )
 *
 * y += alpha * A * x (Hermitian SpMV, CSR full storage, 无清零)
 *
 * 算法结构:
 *   外层循环: 逐行处理 i = 0..dim-1
 *   内层循环: 向量化处理第 i 行的非零元素 j = row_ptr[i]..row_ptr[i+1]-1
 *
 *   上三角 (col >= i) -- 内积操作:
 *     y[i] += alpha * a * x[col]
 *     归约累加,行结束后乘 alpha 写回 y[i]
 *
 *   严格上三角 (col > i) -- 外积操作(逻辑下三角贡献):
 *     y[col] += alpha * conj(a) * x[i]
 *     x[i] 先乘 alpha 再广播,散射累加到 y[col]
 *
 * Register allocation:
 *   x6=i  x7=row_start  x8=row_end  x9=j  x11=VL_doubles
 *   x14=地址计算临时寄存器
 *   x19=y  x20=val  x21=vec  x22=rp  x23=ci  x24=dim  x25=alpha_addr(temp)
 *   q4=vec[i](128bit)  q9=alpha(128bit)
 *   z0=i-bcast  z1=col_idx  z2=val(lo)  z3=val(hi)
 *   z4=x[col](lo)  z5=alpha*x[i]-bcast  z6=x[col](hi)/temp  z7=result(hi)
 *   z8=col*2  z10=temp  z11=内积高半累加器  z12=alpha*x[i] temp
 *   z14=temp  z15=内积低半累加器
 *   p0=loop  p1=col>=i  p2=col>i
 *   p3=zip1(p1)  p4=zip1(p2)  p5=zip2(p1)  p6=zip2(p2)  p7=all-true/reduce
 */

spmv_standard:
	stp x19, x20, [sp, #-96]!
	stp x21, x22, [sp, #16]
	stp x23, x24, [sp, #32]
	stp x29, x30, [sp, #48]
	stp x14, x25, [sp, #64]
	stp d0, d1, [sp, #80]

	mov x19, x0
	mov x20, x1
	mov x21, x2
	mov x22, x3
	mov x23, x4
	mov x24, x5

	ldr q9, [sp, #80]

	rdvl x11, #1
	lsr x11, x11, #3

	index z0.d, #0, #1
	and z0.d, z0.d, #1

	mov x6, #0
3:
	cmp x6, x24
	bge 4f

	add x14, x22, x6, lsl #3
	ldr x7, [x14]
	ldr x8, [x14, #8]

	add x14, x21, x6, lsl #4
	ldr q4, [x14]
	mov z12.d, #0
	fcmla v12.2d, v4.2d, v9.2d, #0
	fcmla v12.2d, v4.2d, v9.2d, #90
	mov z5.q, q12

	mov z15.d, #0
	mov z11.d, #0
	mov x9, x7
5:
	cmp x9, x8
	bge 6f

	add x14, x23, x9, lsl #2
	whilelt p0.d, x9, x8
	ld1sw z1.d, p0/z, [x14]
	add x14, x20, x9, lsl #4
	dup z0.d, x6
	cmpge p1.d, p0/z, z1.d, z0.d
	cmpgt p2.d, p0/z, z1.d, z0.d
	zip1 p3.d, p1.d, p1.d
	zip2 p5.d, p1.d, p1.d
	zip1 p4.d, p2.d, p2.d
	zip2 p6.d, p2.d, p2.d
	ld1d z2.d, p3/z, [x14]
	add x14, x14, x11, lsl #3
	ld1d z3.d, p5/z, [x14]
	lsl z8.d, z1.d, #1
	add x14, x21, #8
	ld1d z4.d, p1/z, [x21, z8.d, lsl #3]
	ld1d z6.d, p1/z, [x14, z8.d, lsl #3]
	zip1 z10.d, z4.d, z6.d
	zip2 z4.d, z4.d, z6.d
	fcmla z15.d, p3/m, z2.d, z10.d, #0
	fcmla z15.d, p3/m, z2.d, z10.d, #90
	fcmla z11.d, p5/m, z3.d, z4.d, #0
	fcmla z11.d, p5/m, z3.d, z4.d, #90
	ld1d z6.d, p2/z, [x19, z8.d, lsl #3]
	add x14, x19, #8
	ld1d z7.d, p2/z, [x14, z8.d, lsl #3]
	zip1 z14.d, z6.d, z7.d
	fcmla z14.d, p4/m, z2.d, z5.d, #0
	fcmla z14.d, p4/m, z2.d, z5.d, #270
	zip2 z10.d, z6.d, z7.d
	fcmla z10.d, p6/m, z3.d, z5.d, #0
	fcmla z10.d, p6/m, z3.d, z5.d, #270
	uzp1 z6.d, z14.d, z10.d
	st1d z6.d, p2, [x19, z8.d, lsl #3]
	uzp2 z7.d, z14.d, z10.d
	st1d z7.d, p2, [x14, z8.d, lsl #3]
	add x9, x9, x11
	b 5b
6:
	ptrue p7.b
	uzp1 z6.d, z15.d, z11.d
	uzp2 z7.d, z15.d, z11.d
	faddv d0, p7, z6.d
	faddv d1, p7, z7.d
	mov v0.d[1], v1.d[0]
	add x14, x19, x6, lsl #4
	ldr q1, [x14]
	fcmla v1.2d, v0.2d, v9.2d, #0
	fcmla v1.2d, v0.2d, v9.2d, #90
	str q1, [x14]
	add x6, x6, #1
	b 3b
4:
	ldp x14, x25, [sp, #64]
	ldp x29, x30, [sp, #48]
	ldp x23, x24, [sp, #32]
	ldp x21, x22, [sp, #16]
	ldp x19, x20, [sp], #96
	ret

	.size spmv_standard, .-spmv_standard
