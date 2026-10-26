#include "../cuda_utils.h"
#include "grouping_cuda_kernel.h"


__global__ void grouping_forward_cuda_kernel(int m, int nsample, int c, const float *__restrict__ input, const int *__restrict__ idx, float *__restrict__ output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * nsample * c) return;
    const int c_idx = index % c;
    const int nsample_idx = (index / c) % nsample;
    const int m_idx = index / nsample / c;
    const int input_idx = idx[m_idx * nsample + nsample_idx] * c + c_idx;
    output[index] = input[input_idx];
}

__global__ void grouping_backward_cuda_kernel(int m, int nsample, int c, const float *__restrict__ grad_output, const int *__restrict__ idx, float *__restrict__ grad_input) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * nsample * c) return;
    const int c_idx = index % c;
    const int nsample_idx = (index / c) % nsample;
    const int m_idx = index / nsample / c;
    const int input_idx = idx[m_idx * nsample + nsample_idx] * c + c_idx;
    atomicAdd(grad_input + input_idx, grad_output[index]);
}

void grouping_forward_cuda_launcher(int m, int nsample, int c, const float *input, const int *idx, float *output) {

    dim3 blocks(DIVUP(m * nsample * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    grouping_forward_cuda_kernel<<<blocks, threads, 0>>>(m, nsample, c, input, idx, output);
}

void grouping_backward_cuda_launcher(int m, int nsample, int c, const float *grad_output, const int *idx, float *grad_input)
{

    dim3 blocks(DIVUP(m * nsample * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    grouping_backward_cuda_kernel<<<blocks, threads, 0>>>(m, nsample, c, grad_output, idx, grad_input);
}
