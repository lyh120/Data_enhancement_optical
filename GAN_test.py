import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

class ConditionalGenerator(nn.Module):
    """条件生成器：根据12维输入生成5维输出"""
    def __init__(self, input_dim=12, noise_dim=50, output_dim=5):
        super(ConditionalGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        
        # 更深的网络结构以学习复杂映射
        self.model = nn.Sequential(
            nn.Linear(input_dim + noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, output_dim)
        )
        
    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)

class ConditionalDiscriminator(nn.Module):
    """条件判别器：判断输入-输出对是否真实"""
    def __init__(self, input_dim=12, output_dim=5):
        super(ConditionalDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + output_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, condition, output):
        x = torch.cat([condition, output], dim=1)
        return self.model(x)

class GANTrainer:
    """GAN离线训练器"""
    def __init__(self, input_dim=12, output_dim=5, noise_dim=50, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.noise_dim = noise_dim
        
        # 初始化网络
        self.generator = ConditionalGenerator(input_dim, noise_dim, output_dim).to(self.device)
        self.discriminator = ConditionalDiscriminator(input_dim, output_dim).to(self.device)
        
        # 初始化优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 数据标准化器
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 训练历史
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
        
    def prepare_data(self, X, y, batch_size=64, validation_split=0.1):
        """准备训练数据"""
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 划分训练集和验证集
        n_samples = len(X_scaled)
        n_train = int(n_samples * (1 - validation_split))
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        # 创建数据集
        train_dataset = TensorDataset(
            torch.FloatTensor(X_scaled[train_idx]),
            torch.FloatTensor(y_scaled[train_idx])
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_scaled[val_idx]),
            torch.FloatTensor(y_scaled[val_idx])
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_step(self, real_inputs, real_outputs):
        """单步训练"""
        batch_size = real_inputs.size(0)
        
        # 标签平滑（提高训练稳定性）
        real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
        
        # ========== 训练判别器 ==========
        self.d_optimizer.zero_grad()
        
        # 真实数据
        real_validity = self.discriminator(real_inputs, real_outputs)
        d_real_loss = self.criterion(real_validity, real_labels)
        
        # 生成假数据
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_outputs = self.generator(noise, real_inputs)
        
        # 假数据
        fake_validity = self.discriminator(real_inputs, fake_outputs.detach())
        d_fake_loss = self.criterion(fake_validity, fake_labels)
        
        # 总判别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.d_optimizer.step()
        
        # ========== 训练生成器 ==========
        self.g_optimizer.zero_grad()
        
        # 生成新的假数据
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_outputs = self.generator(noise, real_inputs)
        validity = self.discriminator(real_inputs, fake_outputs)
        
        # 生成器损失
        g_loss = self.criterion(validity, real_labels)  # 希望判别器认为是真的
        g_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.g_optimizer.step()
        
        # 计算准确率
        real_acc = (real_validity > 0.5).float().mean().item()
        fake_acc = (fake_validity < 0.5).float().mean().item()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_real_acc': real_acc,
            'd_fake_acc': fake_acc
        }
    
    def validate(self, val_loader):
        """验证生成数据质量"""
        self.generator.eval()
        self.discriminator.eval()
        
        mse_losses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 生成多个样本并取平均
                generated_samples = []
                for _ in range(10):
                    noise = torch.randn(inputs.size(0), self.noise_dim).to(self.device)
                    fake_outputs = self.generator(noise, inputs)
                    generated_samples.append(fake_outputs)
                
                avg_generated = torch.stack(generated_samples).mean(dim=0)
                mse_loss = nn.MSELoss()(avg_generated, targets)
                mse_losses.append(mse_loss.item())
        
        self.generator.train()
        self.discriminator.train()
        
        return np.mean(mse_losses)
    
    def train(self, X, y, epochs=200, batch_size=64, save_interval=50):
        """完整训练流程"""
        print(f"开始训练GAN模型...")
        print(f"设备: {self.device}")
        print(f"数据形状: X={X.shape}, y={y.shape}")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data(X, y, batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_metrics = {
                'd_loss': [],
                'g_loss': [],
                'd_real_acc': [],
                'd_fake_acc': []
            }
            
            # 训练一个epoch
            for batch_inputs, batch_outputs in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_outputs = batch_outputs.to(self.device)
                
                metrics = self.train_step(batch_inputs, batch_outputs)
                
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
            
            # 记录平均指标
            for key in epoch_metrics:
                avg_value = np.mean(epoch_metrics[key])
                self.history[key].append(avg_value)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"D Loss: {self.history['d_loss'][-1]:.4f}, "
                      f"G Loss: {self.history['g_loss'][-1]:.4f}, "
                      f"D Real Acc: {self.history['d_real_acc'][-1]:.2%}, "
                      f"D Fake Acc: {self.history['d_fake_acc'][-1]:.2%}, "
                      f"Val MSE: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_gan_model.pth')
            
            # 定期保存
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'gan_checkpoint_epoch_{epoch+1}.pth')
        
        print(f"训练完成！最佳验证MSE: {best_val_loss:.4f}")
        
        # 绘制训练历史
        self.plot_training_history()
        
        return self
    
    def save_checkpoint(self, filepath):
        """保存模型检查点"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'noise_dim': self.noise_dim,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f"模型已保存至: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.noise_dim = checkpoint['noise_dim']
        self.history = checkpoint['history']
        
        print(f"模型已从 {filepath} 加载")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['g_loss'], label='Generator Loss')
        axes[0, 0].plot(self.history['d_loss'], label='Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 判别器准确率
        axes[0, 1].plot(self.history['d_real_acc'], label='Real Data Accuracy')
        axes[0, 1].plot(self.history['d_fake_acc'], label='Fake Data Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 损失比率
        g_d_ratio = np.array(self.history['g_loss']) / (np.array(self.history['d_loss']) + 1e-8)
        axes[1, 0].plot(g_d_ratio)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('G/D Loss Ratio')
        axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
        axes[1, 0].grid(True)
        
        # 移动平均
        window_size = 10
        if len(self.history['g_loss']) >= window_size:
            g_loss_ma = np.convolve(self.history['g_loss'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            d_loss_ma = np.convolve(self.history['d_loss'], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            
            axes[1, 1].plot(g_loss_ma, label=f'G Loss (MA-{window_size})')
            axes[1, 1].plot(d_loss_ma, label=f'D Loss (MA-{window_size})')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Smoothed Training Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('gan_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def train_gan_offline():
    """离线训练GAN的主函数"""
    # 加载离线数据（这里需要替换为实际的数据加载代码）
    # 假设数据已经准备好
    print("加载离线数据...")
    
    # 示例：从文件加载数据
    # data = np.load('offline_data.npz')
    # X = data['settings']  # shape: (n_samples, 12)
    # y = data['qs']        # shape: (n_samples, 5)
    
    # 模拟数据（实际使用时替换）
    n_samples = 10000
    X = np.random.randn(n_samples, 12)
    y = np.random.randn(n_samples, 5)
    
    # 创建训练器
    trainer = GANTrainer(input_dim=12, output_dim=5, noise_dim=50)
    
    # 训练模型
    trainer.train(X, y, epochs=200, batch_size=64, save_interval=50)
    
    # 保存最终模型
    trainer.save_checkpoint('final_gan_model.pth')
    
    print("离线GAN训练完成！")

if __name__ == "__main__":
    train_gan_offline()
