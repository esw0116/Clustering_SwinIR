#include <torch/extension.h>
#include "svf_cuda_kernel.cuh"

void TFGather2D(
    const int b,
    const int l,
    const int k,
    const torch::Tensor labels,
    torch::Tensor attn_labels
)
{
    TFGather2DKernel(
        b, l, k, labels.data_ptr<int64_t>(), attn_labels.data_ptr<int64_t>()
    );
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_2d", &TFGather2D, "TF_gather_2d");
}
