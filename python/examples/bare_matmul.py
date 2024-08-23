import torch

import triton
import triton.language as tl
import benchmark


@triton.jit
def bare_matmul(X, Y, Z, M, N, K, BLOCK_SIZE: tl.constexpr):
    # Get the program IDs for the current block
    pid_x = tl.program_id(0)  # block row id
    pid_y = tl.program_id(1)  # block column id

    # Define offsets for loading submatrices (blocks)
    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load blocks (submatrices) from global memory
    x = tl.load(X + offs_x[:, None] * K + offs_y[None, :])
    y = tl.load(Y + offs_x[:, None] * N + offs_y[None, :])

    # Perform the dot product
    z = tl.dot(x, y)

    # Store the result
    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z)


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
        bare_matmul[(1,)](a, b, c, M, N, K, N) # we assume M == N == K
        print(c)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for X in [2**i for i in range(7, 11, 1)]:
        for provider in ['torch', 'triton']:
            bench_matmul(X, X, X, provider)
