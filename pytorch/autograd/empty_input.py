import torch

"""
Using uninitialized torch.empty() and then doing matmul → Bad (garbage values)
Using torch.empty(0, dim) to represent "no tokens" in MoE setting → okay (valid empty set)

A shape with no data (one dim is 0), not a container of garbage values
As long as the operation:
    never indexes into it
    only reasons about shape
it’s safe.

In backward, we have to preserve shape.
1. weights grad will always have the same shape with weights. Even empty input, it will produce zero grad. There will be 
no issue with ar etc. 

Forward
x.shape      = (0, Dim)
W.shape      = (Dim, Out)
y = x @ W    # (0, Out)

Backward
grad_W = x.T @ grad_y
x.T     : (Dim, 0)
grad_y  : (0, Out)
result  : (Dim, Out)
"""


class SafeLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        """
        x: [N, D]
        weight: [D, O]
        bias: [O]
        """
        ctx.save_for_backward(x, weight, bias)

        # Handle empty input (e.g. MoE expert receives zero tokens)
        if x.numel() == 0:
            return x.new_empty((0, weight.shape[1]))

        return x @ weight + bias

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors

        # Handle empty backward
        if grad_out.numel() == 0:
            grad_x = x.new_empty(x.shape)
            grad_w = weight.new_empty(weight.shape)
            grad_b = bias.new_empty(bias.shape)
            return grad_x, grad_w, grad_b

        grad_x = grad_out @ weight.t()
        grad_w = x.t() @ grad_out
        grad_b = grad_out.sum(dim=0)

        return grad_x, grad_w, grad_b


class SafeLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return SafeLinearFn.apply(x, self.weight, self.bias)


# ------------------------------
# Test: expert receives zero tokens
# ------------------------------

x_empty = torch.empty(0, 4, requires_grad=True)
layer = SafeLinear(4, 8)

y = layer(x_empty)
print("Output shape:", y.shape)

loss = y.sum()
loss.backward()

print("Grad x:", x_empty.grad)
print("Grad weight shape:", layer.weight.grad.shape)
print("Grad bias shape:", layer.bias.grad.shape)
