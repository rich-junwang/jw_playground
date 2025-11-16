import torch
import torch.nn.functional as F

import triton
import triton.language as tl



def eager_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=-1, keepdim=True)[0]
    numerator = torch.exp(x - x_max)
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator


sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype=torch.float32)

print(eager_softmax(sample))
print(F.softmax(sample, dim=-1))
