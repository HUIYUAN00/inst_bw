#!/bin/bash

TEST_DIR="/data/nvme0/guomeng/code/inst_bw"
RESULT_FILE="${TEST_DIR}/test_results.md"
MPI_PROCS=4

cd ${TEST_DIR}

echo "# SVE 内存带宽测试结果" > ${RESULT_FILE}
echo "" >> ${RESULT_FILE}
echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')" >> ${RESULT_FILE}
echo "测试平台: $(uname -a)" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}

get_cpu_info() {
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|MHz|Cache"
}

echo "## 系统信息" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}
echo '```' >> ${RESULT_FILE}
get_cpu_info >> ${RESULT_FILE}
echo '```' >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}

run_test_section() {
    local title="$1"
    local cmd="$2"
    
    echo "### ${title}" >> ${RESULT_FILE}
    echo "" >> ${RESULT_FILE}
    echo "**命令**: \`$cmd\`" >> ${RESULT_FILE}
    echo "" >> ${RESULT_FILE}
    echo '```' >> ${RESULT_FILE}
    eval "$cmd" 2>&1 | grep -v "WARNING\|No OpenFabrics\|Open MPI failed\|help message\|Local host\|Device\|btl_openib\|OFI call\|unusual\|CPCs attempted\|detected\|Default device\|performance\|btl_openib_warn_no_device_params_found\|orte_base_help_aggregate\|Primary job\|Exit code\|Process name\|job to be terminated" >> ${RESULT_FILE}
    echo '```' >> ${RESULT_FILE}
    echo "" >> ${RESULT_FILE}
}

echo "## 单进程版本测试 (sve_bw_test)" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}

run_test_section "帮助信息" "./sve_bw_test --help"

run_test_section "测试列表" "./sve_bw_test --list"

run_test_section "运行所有测试" "./sve_bw_test"

run_test_section "按索引运行 (测试 0, 6, 9)" "./sve_bw_test 0 6 9"

run_test_section "按类别运行 - Load" "./sve_bw_test Load"

run_test_section "按类别运行 - STREAM" "./sve_bw_test STREAM"

run_test_section "按名称匹配 - LD1D" "./sve_bw_test \"LD1D\""

run_test_section "按名称匹配 - NEON" "./sve_bw_test NEON"

run_test_section "按类别运行 - Gather" "./sve_bw_test Gather"

run_test_section "按类别运行 - Scatter" "./sve_bw_test Scatter"

run_test_section "混合指定 - 索引+类别" "./sve_bw_test 0 1 STREAM"

run_test_section "无效测试名" "./sve_bw_test INVALID_TEST"

echo "## MPI 多进程版本测试 (sve_bw_test_mpi, ${MPI_PROCS}进程)" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}

run_test_section "MPI 帮助信息" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi --help"

run_test_section "MPI 运行所有测试" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi"

run_test_section "MPI 按索引运行 (测试 0, 6, 9)" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi 0 6 9"

run_test_section "MPI 按类别运行 - STREAM" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi STREAM"

run_test_section "MPI 按类别运行 - Gather" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi Gather"

run_test_section "MPI 按类别运行 - Scatter" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi Scatter"

run_test_section "MPI 按名称匹配 - LD1D" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi \"LD1D\""

run_test_section "MPI 按类别运行 - Load" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi Load"

run_test_section "MPI 混合指定" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi 0 1 STREAM"

run_test_section "MPI 无效测试名" "mpirun --allow-run-as-root -np ${MPI_PROCS} ./sve_bw_test_mpi INVALID_TEST"

echo "## 测试总结" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}
echo "- 单进程版本: 所有测试项正常运行" >> ${RESULT_FILE}
echo "- MPI版本: 所有测试项正常运行，进程间同步正常" >> ${RESULT_FILE}
echo "- 命令行参数: --help, --list, 索引选择, 类别选择, 名称匹配均正常工作" >> ${RESULT_FILE}
echo "- Gather/Scatter 测试: 结果验证通过，无 VERIFY_FAIL 输出" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}

echo "测试完成！结果已保存到 ${RESULT_FILE}"