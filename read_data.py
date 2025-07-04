import torch

# 加载保存的文件
loaded_data = torch.load('augmented_100data_original.pth')

# 查看输入数据（张量）
inputs = loaded_data['inputs_original']
print("输入数据形状:", inputs.shape)  # 输出类似 torch.Size([100, 12])
print("输入数据示例:", inputs[:2])   # 打印前2个样本的输入特征

# 查看输出数据（张量）
outputs = loaded_data['outputs_original']
print("输出数据形状:", outputs.shape)  # 输出类似 torch.Size([100, 5])
print("输出数据示例:", outputs[:2])   # 打印前2个样本的输出目标
