import torch
import torch.nn as nn

# Notice that in PyTorch we do y = xW.
num_features = 5
fc_param = nn.Parameter(torch.ones(5, 4))
print(fc_param.data.shape)  # 5, 4

# input_feature, output_feature
fc = nn.Linear(5, 4)
print(fc.weight.data.shape)  # 4, 5

data = torch.rand((3, 5))
print("fc layer output shape", fc(data).shape)

# fc_param is just a torch.nn.Parameter, which is essentially a Tensor
# with some extra bookkeeping so it can be tracked by optimizers.
# It’s not a callable nn.Module like nn.Linear, so you can’t do fc_param(data)
print((data @ fc_param).shape)


fc_empty = torch.empty(5, 4)
print("empty tensor shape", fc_empty.shape)


# for nn.Parameter, tensor.data, torch.empty
# when we define a shape, it's effectively defining tensor shape.
# notice that this tensor shape is the transpose of nn.Linear shape.
# This means that usually you'll see that using nn.Parameter, people do
# weight_tensor = torch.empty(output_features, input_features)
# or
# weight = torch.nn.Parameter(torch.empty(output_features, input_features))
# or
# fc = nn.Linear(input_features, output_features)




# gradient

# Parameter
fc_param = nn.Parameter(torch.ones(5, 4))

# Some fake data and target
data = torch.rand((3, 5))
target = torch.rand((3, 4))

# Forward
out = data @ fc_param

# Compute a loss
loss = ((out - target) ** 2).mean()

# Backward
loss.backward()
print(fc_param.grad.shape)  # torch.Size([5, 4])

