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
def bench_matmul(N, provider):
    device = 'cpu'
    dtype = torch.float32
    a = torch.randn((N, N), device=device, dtype=dtype)
    b = torch.randn((N, N), device=device, dtype=dtype)
    c = torch.empty((N, N), device=device, dtype=dtype)
    if provider == 'torch' or provider == 'test':
        c_ref = torch.matmul(a, b)
    if provider == 'triton' or provider == 'test':
        bare_matmul[(1,)](a, b, c, N, N, N, N)
        if provider == 'test':
            torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    bench_matmul(128, 'test')
    for X in [2**i for i in range(7, 10, 1)]:
        for provider in ['torch', 'triton']:
            bench_matmul(X, X, X, provider)
