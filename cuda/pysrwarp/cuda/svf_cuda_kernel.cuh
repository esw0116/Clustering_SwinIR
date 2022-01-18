#ifndef SVF_CUDA_KERNEL
#define SVF_CUDA_KERNEL

void TFGather2DKernel(
    const int b,
    const int l,
    const int k,
    const int64_t* labels,
    int64_t* attn_labels
);

/*
// Depthwise version
void SvfBackwardDCuda(
    const float* x,
    float* dx,
    const float* weight,
    float* dweight,
    const float* dy,
    const int b,
    const int c,
    const int h,
    const int w,
    const int hh,
    const int ww,
    const int k,
    const int* xi,
    const int* yi,
    const int n
);
*/
#endif