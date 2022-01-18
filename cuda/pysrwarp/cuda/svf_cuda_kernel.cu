#ifndef NUM_THREADS
#define NUM_THREADS 1024
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include "svf_cuda_kernel.cuh"


__global__ void TFGather2DCudaKernel(
    const int b,
    const int l,
    const int k,
    const int64_t* labels,
    int64_t* attn_labels
)
{
    // Batch index
    int idx = blockIdx.x;
    // 1st HW index
    int jdx = blockIdx.y;
    // 2nd HW index
    int kdx = threadIdx.x;

    __shared__ int sm[256];
    sm[kdx] = labels[(l * idx) + kdx];
    __syncthreads();

    int idx_attn_labels = (l * l * idx) + (l * jdx) + kdx;

    //int class_ij = labels[(l * idx) + jdx];
    //int class_ik = labels[(l * idx) + kdx];
    int class_ij = sm[jdx];
    int class_ik = sm[kdx];

    attn_labels[idx_attn_labels] = (k * class_ij) + class_ik;
    //attn_labels[idx_attn_labels] = (l * l * idx) + jdx;
    return;
}

void TFGather2DKernel(
    const int b,
    const int l,
    const int k,
    const int64_t* labels,
    int64_t* attn_labels
)
{
    dim3 num_blocks(b, l);
    TFGather2DCudaKernel<<<num_blocks, l>>>(b, l, k, labels, attn_labels);
    return;
}