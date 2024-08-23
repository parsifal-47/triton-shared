import torch

import triton
import triton.language as tl
import benchmark


@triton.jit
def bare_matmul(X, Y, Z, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offs_x = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offs_x[:, None])
    y = tl.load(Y + offs_y[None, :])

    z = tl.dot(x, y)
    tl.store(Z + offs_x[:, None] + offs_y[None, :], z)


@benchmark.measure()
def bench_matmul(M, N, K, provider):
    device = 'cpu'
    dtype = torch.float32
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c = torch.empty((K, N), device=device, dtype=dtype)
    if provider == 'torch':
        c = torch.matmul(a, b)
        print(c)
    if provider == 'triton':
        bare_matmul[(1,)](a, b, c, N)
        print(c)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for X in [2**i for i in range(7, 11, 1)]:
        for provider in ['torch', 'triton']:
            bench_matmul(X, X, X, provider)
