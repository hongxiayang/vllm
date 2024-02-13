#!/bin/bash
#export TENSILE_DB=0x28000
#export HSA_OVERRIDE_GFX_VERSION=9.4.0
for i in 2;
do
export HSA_OVERRIDE_GFX_VERSION=9.4.$i
echo RUNTIME OVERRIDE $HSA_OVERRIDE_GFX_VERSION
HIP_VISIBLE_DEVICuS=0 rocblas-bench --function gemm_ex -m 15360 -n 3072 -k 5120 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transposeA T --transposeB N --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
HIP_VISIBLE_DEVICES=0 rocblas-bench --function gemm_ex -m 5120 -n 3072 -k 5120 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transposeA T --transposeB N --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
HIP_VISIBLE_DEVICES=0 rocblas-bench --function gemm_ex -m 27648 -n 3072 -k 5120 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transposeA T --transposeB N --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
HIP_VISIBLE_DEVICES=0 rocblas-bench --function gemm_ex -m 5120 -n 3072 -k 13824 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transposeA T --transposeB N --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000

HIP_VISIBLE_DEVICES=0 hipblaslt-bench -m 15360 -n 3072 -k 5120 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transA T --transB N --precision f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
HIP_VISIBLE_DEVICES=0 hipblaslt-bench  -m 5120 -n 3072 -k 5120 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transA T --transB N --precision f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
HIP_VISIBLE_DEVICES=0 hipblaslt-bench  -m 27648 -n 3072 -k 5120 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transA T --transB N --precision f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
HIP_VISIBLE_DEVICES=0 hipblaslt-bench  -m 5120 -n 3072 -k 13824 --lda 4864 --ldb 8192 --ldc 4864 --ldd 4864 --transA T --transB N --precision f16_r --compute_type f32_r --alpha 1 --beta 0 --iters 10000 --cold_iters 1000
done