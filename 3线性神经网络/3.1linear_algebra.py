import torch
# 标量
x1= torch.tensor(3.0)
print(x1)
print("---------------")

# 向量
x2 = torch.arange(4)
print(x2)
print(len(x2), x2.shape)
print("---------------")

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)   # 转置

# 张量
A2 = torch.arange(24).reshape(2, 3, 4)
print(A2)

