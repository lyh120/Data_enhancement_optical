# -*- coding: utf-8 -*-
"""
使用条件Wasserstein GAN with Gradient Penalty (cWGAN-GP) 和谱归一化 (Spectral Normalization)
进行数据增强的完整Python脚本。

任务:
- 输入 (条件 Y): 5个Q-factor值。
- 输出 (生成 X): 12个参数 (6个增益G, 6个倾斜度T)。
- 目标: 为特定的工程问题生成高质量的增强数据。

结构:
1. ConfigManager: 管理所有超参数。
2. DataPreprocessor: 加载、标准化和逆标准化数据。
3. Generator: cGAN的生成器网络。
4. Discriminator: cGAN的判别器/评论家网络，使用谱归一化。
5. ModelTrainer: 封装cWGAN-GP训练循环、梯度惩罚和模型保存。
6. 主程序:
   - 创建模拟数据以保证脚本可独立运行。
   - 协调数据准备、模型训练流程。
   - 生成增强数据。
   - 通过分布图和相关性热图验证数据质量。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# 设置Matplotlib以正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 1. 配置管理器
class ConfigManager:
    """统一管理所有超参数"""
    def __init__(self):
        self.data_path = 'simulated_data.npz'  # 数据文件路径
        self.model_save_dir = 'model_checkpoints' # 模型保存目录
        self.n_epochs = 3000                 # 训练轮数
        self.batch_size = 64                 # 批处理大小
        self.lr_g = 0.0001                   # 生成器学习率
        self.lr_d = 0.0001                   # 判别器学习率
        self.b1 = 0.5                        # Adam优化器参数
        self.b2 = 0.999                      # Adam优化器参数
        self.latent_dim = 128                # 噪声向量维度
        self.input_dim = 12                  # 数据X的维度 (6个增益, 6个倾斜度)
        self.condition_dim = 5               # 条件Y的维度 (5个Q-factors)
        self.n_critic = 5                    # 每训练一次生成器，训练判别器的次数
        self.lambda_gp = 10                  # 梯度惩罚的系数
        self.sample_interval = 500           # 每隔多少轮次保存一次模型

        # 12个特征的物理有效范围
        self.physical_bounds = {
            # 增益 (Gain)
            'G0': (14, 23), 'G1': (14, 23), 'G2': (14, 23),
            'G3': (14, 23), 'G4': (14, 23), 'G5': (14, 23),
            # 倾斜度 (Tilt)
            'T0': (-10, 10), 'T1': (-10, 10), 'T2': (-10, 10),
            'T3': (-10, 10), 'T4': (-10, 10), 'T5': (-10, 10),
        }
        self.feature_names = list(self.physical_bounds.keys())

# 2. 数据预处理器
class DataPreprocessor:
    """负责加载、标准化和逆标准化数据"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.X = None
        self.Y = None

    def load_data(self):
        """从 .npz 文件加载数据"""
        try:
            data = np.load(self.filepath)
            self.X = data['X']
            self.Y = data['Y']
            print(f"成功从 '{self.filepath}' 加载数据。")
            print(f"  - 特征数据 X 的形状: {self.X.shape}")
            print(f"  - 条件数据 Y 的形状: {self.Y.shape}")
        except FileNotFoundError:
            print(f"错误: 数据文件 '{self.filepath}' 未找到。请先生成模拟数据。")
            return False
        return True

    def preprocess(self):
        """对特征数据X进行标准化"""
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.X_scaled, self.Y

    def inverse_transform(self, X_scaled):
        """将标准化后的数据恢复到原始尺度"""
        return self.scaler.inverse_transform(X_scaled)

# 3. 模型定义
class Generator(nn.Module):
    """
    生成器网络 (Generator)
    输入: 噪声向量 + 条件向量Y
    输出: 12维数据样本X
    """
    def __init__(self, config):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config.latent_dim + config.condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.input_dim),
            nn.Tanh() # 使用Tanh确保输出在[-1, 1]范围内, 与StandardScaler的特性匹配
        )

    def forward(self, noise, conditions):
        # 将噪声和条件向量拼接作为输入
        gen_input = torch.cat((noise, conditions), -1)
        return self.model(gen_input)

class Discriminator(nn.Module):
    """
    判别器/评论家网络 (Discriminator/Critic)
    输入: 12维数据样本X + 条件向量Y
    输出: 一个标量值 (Wasserstein距离的估计)
    特点: 在所有线性层（除输出层外）后使用谱归一化。
    """
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # `torch.nn.utils.spectral_norm` 用于增强训练稳定性
            nn.utils.spectral_norm(nn.Linear(config.input_dim + config.condition_dim, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出层是一个标量，没有激活函数
            nn.Linear(256, 1)
        )

    def forward(self, data, conditions):
        # 将数据和条件向量拼接作为输入
        disc_input = torch.cat((data, conditions), -1)
        return self.model(disc_input)

# 4. 模型训练器
class ModelTrainer:
    """封装cWGAN-GP的完整训练逻辑"""
    def __init__(self, generator, discriminator, config, device):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = device
        self._initialize_optimizers()
        os.makedirs(self.config.model_save_dir, exist_ok=True)

    def _initialize_optimizers(self):
        """初始化生成器和判别器的优化器"""
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=self.config.lr_g, betas=(self.config.b1, self.config.b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config.lr_d, betas=(self.config.b1, self.config.b2)
        )

    def _compute_gradient_penalty(self, real_samples, fake_samples, conditions):
        """计算梯度惩罚 (Gradient Penalty)"""
        # 随机选择插值点
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        
        # 创建插值样本
        interpolated = (alpha * real_samples.data + (1 - alpha) * fake_samples.data).requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        prob_interpolated = self.discriminator(interpolated, conditions)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # 计算梯度的L2范数并惩罚
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, dataloader):
        """执行完整的训练循环"""
        print("\n--- 开始模型训练 ---")
        for epoch in range(self.config.n_epochs):
            for i, (real_data_X, real_data_Y) in enumerate(dataloader):
                
                real_X = real_data_X.to(self.device).float()
                conditions = real_data_Y.to(self.device).float()
                batch_size = real_X.size(0)

                # ---------------------
                #  训练判别器 (Critic)
                # ---------------------
                self.optimizer_D.zero_grad()

                # 生成噪声向量
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                
                # 生成假数据
                fake_X = self.generator(z, conditions).detach()

                # 计算判别器对真假数据的输出
                real_validity = self.discriminator(real_X, conditions)
                fake_validity = self.discriminator(fake_X, conditions)
                
                # 计算梯度惩罚
                gradient_penalty = self._compute_gradient_penalty(real_X, fake_X, conditions)
                
                # 判别器损失 = Wasserstein距离 + 梯度惩罚
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.config.lambda_gp * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                # 每 n_critic 次迭代后训练一次生成器
                if i % self.config.n_critic == 0:
                    # -----------------
                    #  训练生成器
                    # -----------------
                    self.optimizer_G.zero_grad()

                    # 生成新的假数据
                    z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                    gen_X = self.generator(z, conditions)
                    
                    # 计算判别器对新假数据的输出
                    gen_validity = self.discriminator(gen_X, conditions)
                    
                    # 生成器损失
                    g_loss = -torch.mean(gen_validity)

                    g_loss.backward()
                    self.optimizer_G.step()
            
            # --- 打印训练进度并保存模型 ---
            if (epoch + 1) % 100 == 0:
                print(
                    f"[轮次 {epoch+1}/{self.config.n_epochs}] "
                    f"[D损失: {d_loss.item():.4f}] "
                    f"[G损失: {g_loss.item():.4f}]"
                )

            if (epoch + 1) % self.config.sample_interval == 0:
                self.save_models(epoch + 1)
        
        print("--- 模型训练完成 ---")

    def save_models(self, epoch):
        """保存模型检查点"""
        g_path = os.path.join(self.config.model_save_dir, f"generator_epoch_{epoch}.pth")
        d_path = os.path.join(self.config.model_save_dir, f"discriminator_epoch_{epoch}.pth")
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)
        print(f"已在轮次 {epoch} 保存模型检查点。")

# 5. 辅助函数
def create_simulated_data(config, num_samples=2000):
    """
    创建一个模拟的 .npz 数据文件，以确保脚本可以独立运行。
    真实数据中，特征之间可能存在复杂的相关性。
    """
    if os.path.exists(config.data_path):
        print(f"'{config.data_path}' 已存在，跳过创建。")
        return

    print(f"正在创建模拟数据文件 '{config.data_path}'...")
    # 生成5维的条件Y (Q-factors)
    Y = np.random.uniform(low=0.5, high=5.0, size=(num_samples, config.condition_dim))
    
    # 生成12维的特征X (Gains and Tilts)，并引入一些相关性
    # 基础随机数据
    base = np.random.randn(num_samples, config.input_dim)
    
    # 创建一个简单的相关性结构
    X = np.zeros_like(base)
    for i in range(6): # Gains
        # 增益受Q-factor总和的影响
        X[:, i] = np.clip(18.5 + base[:, i] * 2 + np.sum(Y, axis=1) * 0.1, 14, 23)
    for i in range(6, 12): # Tilts
        # 倾斜度受特定Q-factor和噪声影响
        X[:, i] = np.clip(0 + base[:, i] * 4 - Y[:, i-6 % 5] * 0.5, -10, 10)

    # 引入一些特征间的相关性
    X[:, 1] += X[:, 0] * 0.2
    X[:, 7] -= X[:, 6] * 0.3
    
    np.savez(config.data_path, X=X, Y=Y)
    print("模拟数据创建完成。")

def clip_to_physical_bounds(data, config):
    """
    将生成的数据裁剪到其物理有效范围内。
    这是整合领域知识的关键步骤。
    """
    clipped_data = np.zeros_like(data)
    for i, name in enumerate(config.feature_names):
        min_val, max_val = config.physical_bounds[name]
        clipped_data[:, i] = np.clip(data[:, i], min_val, max_val)
    return clipped_data

def generate_augmented_data(generator, preprocessor, condition_vector, config, device, num_samples=1000):
    """
    使用训练好的生成器生成增强数据。
    
    Args:
        generator (nn.Module): 训练好的生成器模型。
        preprocessor (DataPreprocessor): 用于逆标准化的数据处理器。
        condition_vector (np.array): 一个5维的条件向量Y。
        config (ConfigManager): 配置对象。
        device (torch.device): 'cuda' 或 'cpu'。
        num_samples (int): 要生成的样本数量。
        
    Returns:
        np.array: 生成并经过后处理的增强数据。
    """
    print(f"\n--- 为条件 {condition_vector} 生成 {num_samples} 个增强样本 ---")
    generator.eval() # 设置为评估模式
    
    # 准备输入
    z = torch.randn(num_samples, config.latent_dim, device=device)
    conditions = torch.from_numpy(condition_vector).float().unsqueeze(0).repeat(num_samples, 1).to(device)
    
    # 生成标准化尺度的数据
    with torch.no_grad():
        generated_scaled_X = generator(z, conditions).cpu().numpy()
        
    # 1. 逆标准化，恢复到原始数据尺度
    generated_original_X = preprocessor.inverse_transform(generated_scaled_X)
    
    # 2. 裁剪到物理有效范围
    generated_clipped_X = clip_to_physical_bounds(generated_original_X, config)
    
    print("数据增强完成，并已应用物理范围裁剪。")
    return generated_clipped_X


def plot_distribution_comparison(real_data, augmented_data, config, condition_str):
    """
    为每个特征绘制真实数据与增强数据的分布对比图 (KDE)。
    
    这个图表帮助我们判断：
    - 生成数据的分布形状是否与真实数据一致。
    - 生成数据的均值、方差和峰度是否与真实数据相似。
    - 生成的数据是否覆盖了真实数据的主要范围，没有产生模式崩溃（Mode Collapse）。
    """
    print("\n--- 正在生成特征分布对比图 ---")
    num_features = real_data.shape[1]
    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    axes = axes.flatten()

    for i in range(num_features):
        feature_name = config.feature_names[i]
        sns.kdeplot(real_data[:, i], ax=axes[i], label='真实数据', color='blue', fill=True)
        sns.kdeplot(augmented_data[:, i], ax=axes[i], label='增强数据', color='orange', fill=True)
        axes[i].set_title(f'特征 "{feature_name}" 的分布对比')
        axes[i].set_xlabel('值')
        axes[i].set_ylabel('密度')
        axes[i].legend()

    # 隐藏多余的子图
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(f'真实数据与增强数据分布对比\n(条件: {condition_str})', fontsize=24, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data, title, feature_names):
    """
    计算并可视化数据集的特征相关性热图。
    
    这个图表帮助我们判断：
    - 增强数据是否成功学习并复现了原始数据中特征之间的线性关系。
    - 重要的正相关和负相关关系是否被保留。
    - 例如，如果G0和G1在真实数据中强正相关，增强数据也应表现出类似的模式。
    """
    print(f"--- 正在生成 '{title}' 的相关性热图 ---")
    df = pd.DataFrame(data, columns=feature_names)
    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title(title, fontsize=18)
    plt.show()


# 6. 主程序入口
if __name__ == "__main__":
    # --- 1. 初始化 ---
    config = ConfigManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 准备数据 ---
    # 如果没有真实数据，则创建模拟数据
    create_simulated_data(config)
    
    # 加载并预处理数据
    preprocessor = DataPreprocessor(config.data_path)
    if not preprocessor.load_data():
        exit() # 如果数据加载失败则退出
        
    X_scaled, Y = preprocessor.preprocess()
    
    # 创建PyTorch DataLoader
    dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(Y))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # --- 3. 初始化模型和训练器 ---
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    trainer = ModelTrainer(generator, discriminator, config, device)

    # --- 4. 训练模型 ---
    # 注意: GAN的训练需要较长时间才能获得好结果，这里的轮数仅为演示。
    # 对于实际应用，可能需要数万甚至数十万轮。
    # trainer.train(dataloader)

    # --- 5. 数据增强与验证 ---
    # 为了演示，我们将加载一个预训练模型或使用新初始化的模型。
    # 如果没有训练，生成的数据将是随机的。这里我们直接加载最新模型（如果存在）
    try:
        latest_epoch = config.n_epochs # 假设我们训练完成了所有轮次
        gen_model_path = os.path.join(config.model_save_dir, f"generator_epoch_{latest_epoch}.pth")
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        print(f"\n成功加载预训练生成器模型: {gen_model_path}")
    except FileNotFoundError:
        print("\n未找到预训练模型，将使用新初始化的生成器进行演示。")
        print("警告: 未经训练的生成器产生的数据是无意义的随机噪声。")
        # 即使没有模型，也继续执行以展示流程
        trainer.train(dataloader) # 如果没模型就现场训练

    # 选择一个固定的条件向量Y进行数据增强
    # 我们从原始数据中选择第一个样本的条件作为示例
    target_condition = preprocessor.Y[0] 
    condition_str_for_plot = np.array2string(target_condition, formatter={'float_kind': lambda x: "%.2f" % x})


    # 生成增强数据
    augmented_X = generate_augmented_data(generator, preprocessor, target_condition, config, device, num_samples=1000)

    # 获取与该条件对应的所有真实数据，用于对比
    # 注意: 由于Y是连续值，精确匹配可能找不到样本。这里我们寻找最接近的。
    # 为简单起见，我们假设原始数据集中有足够多的样本与target_condition完全相同。
    # 在真实场景中，可能需要按条件对数据进行分箱或聚类。
    # 此处我们用一个简化的方式：随机选择1000个真实样本作为对比基准。
    # 这是一个简化，理想情况下应选择与`target_condition`相似的真实样本。
    indices = np.random.choice(len(preprocessor.X), 1000, replace=False)
    real_X_for_comparison = preprocessor.X[indices]


    print("\n" + "="*50)
    print("数据增强质量验证")
    print("="*50)
    print("接下来的图表将帮助我们评估生成的数据质量。一个好的模型应该：")
    print("1. 生成的数据在每个维度上的分布（KDE图）与真实数据相似。")
    print("2. 生成的数据保留了原始数据中特征之间的相关性（相关性热图）。")
    
    # 验证1: 对比特征分布
    plot_distribution_comparison(real_X_for_comparison, augmented_X, config, condition_str_for_plot)
    
    # 验证2: 对比特征相关性
    plot_correlation_heatmap(real_X_for_comparison, "真实数据的特征相关性热图", config.feature_names)
    plot_correlation_heatmap(augmented_X, "增强数据的特征相关性热图", config.feature_names)
