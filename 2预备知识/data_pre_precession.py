# 数据预处理

import os
import pandas as pd
import torch

# 创建一个人工数据集，并存储在CSV（逗号分隔值）文件
os.makedirs(os.path.join('../..', 'data'), exist_ok=True)
data_file = os.path.join('../..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 从创建的CSV文件中加载原始数据集
data = pd.read_csv(data_file)
print(data)

print("---------------------------------")

# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

print("-----------------------")

# 将数据转化为张量
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)