# 实例化网络  电子书没有pytorch版本  代码来自kimi

import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)  # 假设输入数据的特征维度是20
net(X)

print(net[0].weight.shape)  # 输出应该是torch.Size([256, 20])

print(net[0].weight)  # 在传递数据之前，输出会是<UninitializedParameter>


print(net)