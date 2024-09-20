# 具有单隐藏层的多层感知机
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 参数访问
print(net[2].state_dict())

# 目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# 看看它是怎么工作的
print(rgnet)
# 访问第一个主要的块中、第二个子块的第一层的偏置项。
print(rgnet[0][1][0].bias.data)



#  参数初始化
# 就先不写了， 具体的看电子书