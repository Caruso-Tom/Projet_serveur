import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garr
from pycuda.compiler import SourceModule
cuda.init()
# On choisit le GPU sur lequel le code va tourner, entre 0 et 3
dev = cuda.Device(3)
contx = dev.make_context()

size_x = 64
size_y = 32

c = np.zeros((size_x,size_y))
"""a = np.ones((size_y, size_x))
b = np.ones((size_y, size_x))
a = a.astype(np.float32)
b = b.astype(np.float32)
c = c.astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)"""

a_gpu = garr.ones(size_x*size_y)
b_gpu = garr.ones(size_x*size_y)
c_gpu = garr.zeros(size_x*size_y)

mod = SourceModule("""
    __global__ void add(float *a,float *b,float *c)
    { int idx = threadIdx.x +blockIdx.x*blockDim.x+ (threadIdx.y+blockDim.y * blockIdx.y) * blockDim.x *gridDim.x;
        c[idx] = threadIdx.y*a[idx] + b[idx];
    }
""")

func = mod.get_function("add")
func(a_gpu, b_gpu, c_gpu, block=(32, 32, 1), grid=(2, 1))
cuda.memcpy_dtoh(c, c_gpu)
print("\n", c[:, 1])
print("\n", c[1, :])
contx.pop()
