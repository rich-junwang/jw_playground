# This is only for learning purpose matmul. Not for practical use case.
# This is to illustrate 2D launch grid use case in triton

import torch

import triton
import triton.language as tl

def matmul(X, Y):
    x_rows, x_cols = X.shape
    y_rows, y_cols = Y.shape
    output = torch.empty(x_rows, y_cols, device="cuda") # Output matrix
    
    # Block size is the power of 2 greater than the number of columns in X
    # Multiply by 2 to load both the x row and the y column
    BLOCK_SIZE = triton.next_power_of_2(x_cols)
    
    # Set number of warps higher if we have a higher block size to speed up computation
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Create a 2-D grid to iterate across rows and columns
    grid = lambda meta: (x_rows, y_cols)
    
    Y = Y.T.contiguous() # this call transposes Y, so we can load entire columns at once.  The contiguous call ensures the tensor is reshaped in memory, too.
    matmul_kernel[grid](X, Y, output, x_rows, x_cols, y_rows, y_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    
    return output


@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    x_rows,
    x_cols,
    y_rows,
    y_cols,
    BLOCK_SIZE: tl.constexpr
):
    x_row_id = tl.program_id(axis=0) # x row position for multiplication
    y_col_id = tl.program_id(axis=1) # y column position for multiplication
    
    x_row_start = x_row_id * x_cols # Find the first index of the row we're multiplying in x
    y_col_start = y_col_id * y_rows # find the first index of the col we're multiplying in y
    
    x_row_offset = x_row_start + tl.arange(0, BLOCK_SIZE) # The entire x row pointers
    x_mask = x_row_offset < x_row_start + x_cols # Mask out any x data that is after the last column
    
    y_col_offset = y_col_start + tl.arange(0, BLOCK_SIZE) # The entire y column pointers
    y_mask = y_col_offset < y_col_start + y_rows # Mask out any y data that is after the last row
    
    x_row = tl.load(x_ptr + x_row_offset, mask=x_mask, other=0.0) # Load the row
    y_col = tl.load(y_ptr + y_col_offset, mask=y_mask, other=0.0) # Load the column
    
    output = tl.sum(x_row * y_col, axis=0) # Multiply and sum
     
    output_offset = (x_row_id * y_cols) + y_col_id # We're only storing a single number
    
    tl.store(output_ptr + output_offset, output)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.arange(12, device='cuda', dtype=torch.float32).reshape(4,3)
    y = torch.arange(6, device='cuda', dtype=torch.float32).reshape(3,2)

    output_torch = x @ y
    output_triton = matmul(x, y)
    print(
        f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}'
    )
    print(output_torch[:10])
    print(output_triton[:10])
