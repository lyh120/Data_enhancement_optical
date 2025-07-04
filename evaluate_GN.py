import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 添加文件夹到系统路径
sys.path.append(r".\Raman_simulator_for_LLM\scripts")

# 现在可以导入该文件夹中的模块
from propagation_EDFA_C_420 import gn_occ_new

def GN(setting):
    """使用GN模型计算Q因子"""
    q_factors = gn_occ_new(loading_status=[1,1,1,1,1,0], gain_list=setting[:6], tilt_list=setting[-6:], bool_verbose=False)
    return q_factors[:5]

def evaluate_gn_accuracy(settings, qs, num_samples=20):
    """
    评估GN模型的准确性
    
    参数:
    settings - 设置参数数组
    qs - 真实Q因子数组
    num_samples - 要评估的随机样本数量
    
    返回:
    rmse - 均方根误差
    sample_rmse - 每个样本的RMSE
    component_rmse - 每个Q因子分量的RMSE
    sample_indices - 选择的样本索引
    predicted_qs - GN模型预测的Q因子
    true_qs - 对应的真实Q因子
    all_true - 所有真实值（展平）
    all_pred - 所有预测值（展平）
    """
    # 确保样本数量不超过数据集大小
    num_samples = min(num_samples, len(settings))
    random.seed = 42
    # 随机选择样本索引
    sample_indices = random.sample(range(len(settings)), num_samples)
    
    # 存储预测值和真实值
    predicted_qs = []
    true_qs = []
    
    # 对每个样本进行评估
    for idx in sample_indices:
        setting = settings[idx]
        true_q = qs[idx]
        
        # 使用GN模型预测Q因子
        predicted_q = GN(setting)
        
        predicted_qs.append(predicted_q)
        true_qs.append(true_q)
    
    # 转换为numpy数组以便计算
    predicted_qs = np.array(predicted_qs)
    true_qs = np.array(true_qs)
    
    # 计算每个样本的RMSE
    sample_rmse = []
    for i in range(num_samples):
        rmse = np.sqrt(mean_squared_error(true_qs[i], predicted_qs[i]))
        sample_rmse.append(rmse)
    
    # 计算每个Q因子分量的RMSE
    component_rmse = []
    for i in range(5):  # 5个波长
        component_true = true_qs[:, i]
        component_pred = predicted_qs[:, i]
        rmse = np.sqrt(mean_squared_error(component_true, component_pred))
        component_rmse.append(rmse)
    
    # 计算总体RMSE
    all_true = true_qs.flatten()
    all_pred = predicted_qs.flatten()
    overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    
    return overall_rmse, sample_rmse, component_rmse, sample_indices, predicted_qs, true_qs, all_true, all_pred

# 加载数据集
dataset = np.load('./lhs_dataset_5wave_500.npz')
settings = dataset['settings']
qs = dataset['q_values']
# 计算每个样本的Q因子均值
q_means = np.mean(qs, axis=1)
# 获取Q因子均值排序后的索引（降序）
sorted_indices = np.argsort(q_means)[::-1]
# 选取Q因子均值最高的前450个样本
top_450_indices = sorted_indices[:450]
settings = settings[top_450_indices]
qs = qs[top_450_indices]


# 评估GN模型准确性
overall_rmse, sample_rmse, component_rmse, sample_indices, predicted_qs, true_qs, all_true, all_pred = evaluate_gn_accuracy(settings, qs, num_samples=10)

# 打印结果
print(f"Overall RMSE: {overall_rmse:.4f}")
print(f"Average Sample RMSE: {np.mean(sample_rmse):.4f}")
print(f"Min Sample RMSE: {np.min(sample_rmse):.4f}")
print(f"Max Sample RMSE: {np.max(sample_rmse):.4f}")

# 打印每个Q因子分量的RMSE
print("\nRMSE for each Q-factor component:")
for i, rmse in enumerate(component_rmse):
    print(f"Component {i+1}: {rmse:.4f}")

# 可视化结果
plt.figure(figsize=(15, 10))

# 绘制每个样本的RMSE
plt.subplot(2, 3, 1)
plt.bar(range(len(sample_rmse)), sample_rmse)
plt.xlabel('Sample Index')
plt.ylabel('RMSE')
plt.title('RMSE for Each Sample')

# 绘制每个Q因子分量的RMSE
plt.subplot(2, 3, 2)
plt.bar(range(len(component_rmse)), component_rmse)
plt.xlabel('Q-factor Component')
plt.ylabel('RMSE')
plt.title('RMSE for Each Q-factor Component')

# 绘制预测值与真实值的散点图
plt.subplot(2, 3, 3)
plt.scatter(all_true, all_pred, alpha=0.5)
plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
plt.xlabel('True Q-factor')
plt.ylabel('Predicted Q-factor')
plt.title('Predicted vs True Q-factors')

# 绘制每个波长的Q因子预测误差箱线图
errors = predicted_qs - true_qs
plt.subplot(2, 3, 4)
plt.boxplot([errors[:, i] for i in range(5)])
plt.xlabel('Wavelength Index')
plt.ylabel('Prediction Error')
plt.title('Q-factor Prediction Error by Wavelength')

# 绘制每个样本的真实值与预测值比较
plt.subplot(2, 3, 5)
x = np.arange(5)
width = 0.35
for i in range(min(5, len(sample_indices))):  # 只显示前5个样本
    plt.bar(x - width/2 + i*width/5, true_qs[i], width/5, label=f'True {i}', alpha=0.7)
    plt.bar(x + width/2 + i*width/5, predicted_qs[i], width/5, label=f'Pred {i}', alpha=0.7)
plt.xlabel('Wavelength Index')
plt.ylabel('Q-factor')
plt.title('True vs Predicted Q-factors (First 5 Samples)')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# 绘制每个Q因子分量的散点图
plt.subplot(2, 3, 6)
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(5):
    plt.scatter(true_qs[:, i], predicted_qs[:, i], color=colors[i], alpha=0.7, label=f'Component {i+1}')
plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'k--')
plt.xlabel('True Q-factor')
plt.ylabel('Predicted Q-factor')
plt.title('Component-wise Prediction Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#利用GN模型去预测前450个数据供给训练
qs_gn = []
for idx in range(len(settings)):
    setting = settings[idx]
    q_gn = GN(setting)
    qs_gn.append(q_gn)
    # 打印当前处理进度
    print(f"Processing sample {idx+1}/{len(settings)} ({(idx+1)/len(settings)*100:.1f}%)")
qs_gn = np.array(qs_gn)

# 保存预测结果和对应的settings到npz文件
np.savez(
    './results/gn_prediction_results.npz',  # 保存的文件名
    settings=settings,            # 键名'settings'对应处理后的参数数组
    predicted_qs=qs_gn,   # 键名'predicted_qs'对应模型预测的Q因子数组
)
print("Prediction results and settings saved to './results/gn_prediction_450results.npz'")
