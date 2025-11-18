import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    print("mask", mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# Create sample input tensors
N = 100
x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')
out = torch.empty_like(x)

# Define grid and launch kernel
# notice that here BLOCK_SIZE is set by user. triton.cdiv(n_cols, BLOCK_SIZE) is needed when we split a single row across multiple blocks (i.e., BLOCK_SIZE < n_cols or we deliberately shard columns).

# In softmax and matmul cases, BLOCK_SIZE = triton.next_power_of_2(n_cols), BLOCK_SIZE is always larger than n_cols, so a single block can cover the entire row.
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, out, N, BLOCK_SIZE=32)

# Verify results
expected = x + y
print(f"Results match: {torch.allclose(out, expected)}")
print(f"Max difference: {torch.max(torch.abs(out - expected))}")
