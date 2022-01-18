NVCC = /usr/local/cuda-10.2/bin/nvcc
INCFLAGS = -I /usr/local/cuda-10.2/include
CUDAFLAGS = -shared -O2 -gencode=arch=compute_75,code=sm_75 -std=c++14
CUDAFLAGSADD = --compiler-options "-fPIC"

all: libsvf_cuda_kernel.so
libsvf_cuda_kernel.so: svf_cuda_kernel.cu
	$(NVCC) $(INCFLAGS) -o $@ -c $^ $(CUDAFLAGS) $(CUDAFLAGSADD)

clean:
	rm -f ./*.so*
