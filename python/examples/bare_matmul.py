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

    # Define bounds checking to avoid loading out-of-bounds values
    mask_x = offs_x < M
    mask_y = offs_y < N

    # Load blocks (submatrices) from global memory
    x = tl.load(X + offs_x[:, None] * K + tl.arange(0, K)[None, :], mask=mask_x[:, None])
    y = tl.load(Y + tl.arange(0, K)[:, None] * N + offs_y[None, :], mask=mask_y[None, :])

    # Perform the dot product
    z = tl.dot(x, y)

    # Store the result
    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z, mask=mask_x[:, None] & mask_y[None, :])


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
