import torch

# here quantization we use orig_data = qdata / scale, note sometimes people use orig_data = qdata * scale
t = torch.randn(2, 32, dtype=torch.bfloat16)

row, col = t.shape
block_size = 32

MAX_VALUE = 448.0
t_block = t.view(-1, block_size)
amax = torch.amax(t_block.abs(), dim=-1, keepdim=True)
scale = amax / MAX_VALUE

qdata = t / scale
qdata = torch.clamp(qdata, -MAX_VALUE, MAX_VALUE)
scale = scale.to(torch.float8_e8m0fnu)

