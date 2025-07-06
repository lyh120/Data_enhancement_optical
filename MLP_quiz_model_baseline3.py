import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

class LipschitzNN(nn.Module):
    def __init__(self, input_dim=12, output_dim=5):  # 移除了L_lambda和L_const参数
        super().__init__()
        self.Net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        self.scaler_X = None
        self.scaler_y = None
    
    def forward(self, x):
        return self.Net(x)
    
    def lip_loss(self, x, y_pred):
        sigma_x = torch.FloatTensor(self.scaler_X.scale_)
        sigma_y = torch.FloatTensor(self.scaler_y.scale_)
        Lip_scaled = self.Lip_const * (sigma_x.reshape(1, -1) / sigma_y.reshape(-1, 1))
        batch_size = x.size(0)
        
        jacobian = torch.zeros(batch_size, y_pred.size(1), x.size(1)).to(x.device)
        for i in range(y_pred.size(1)):
            gradients = torch.autograd.grad(
                outputs=y_pred[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]
            jacobian[:, i, :] = gradients
        
        jacobian_abs = torch.abs(jacobian)
        Lip_expanded = Lip_scaled.unsqueeze(0).expand(batch_size, -1, -1)
        excess = (jacobian_abs - Lip_expanded).clamp(min=0)
        return torch.mean(excess**2)
    
def plot_loss_curves(train_losses, val_losses, title="Training Process"):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, title="Prediction Comparison"):
    plt.figure(figsize=(15, 6))
    for i in range(y_true.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel(f'True Q{i+1}')
        plt.ylabel(f'Predicted Q{i+1}')
        plt.title(f'Q{i+1} Prediction (R2={r2_score(y_true[:,i], y_pred[:,i]):.3f})')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# ----->  Offline training                        
def offline_training(num_epoch=500, save_path="enhanced_model.pth"):  # 移除了Lip_lambda和L_const参数
    # # 数据加载与预处理
    # loaded_data = np.load("./new_dataset2.npz")
    # setting_list = loaded_data['setting_list'].reshape(-1, 12)
    # q_list = loaded_data['q_list'].reshape(-1, 6)

    # 数据加载与预处理
    #loaded_data = np.load("./Lip_GN_5wave_0319.npz")
    #加载主训练模型
    loaded_data = np.load('augmented_data_1000_V2.npz')
    setting_list = loaded_data['settings'].reshape(-1, 12)
    q_list = loaded_data['q_values'].reshape(-1, 5)

    

    # 加载独立验证真实数据集
    test_data = np.load("./lhs_dataset_5wave_500.npz")
    X_test_external = test_data['settings'][20:].reshape(-1, 12)
    y_test_external = test_data['q_values'][20:].reshape(-1, 5)

    
    
    #打印检验
    print("原始验证数据集：")
    print(X_test_external.shape)
    print(y_test_external.shape)


    # # 数据增强
    # def add_noise(data, noise_level=0.01):
    #     return data + np.random.normal(0, noise_level, data.shape)
    
    # 扩展数据集
    augmented_settings = np.vstack([setting_list, test_data['settings'][:20].reshape(-1, 12)])
    augmented_q = np.vstack([q_list, test_data['q_values'][:20].reshape(-1, 5)])
    
    # #直接使用原始20组数据进行训练
    # augmented_settings = test_data['settings'][:20].reshape(-1, 12)
    # augmented_q = test_data['q_values'][:20].reshape(-1, 5)

    #把未增强的20个数据集合加入进去
    print("增强数据集：")
    print(augmented_settings.shape)
    print(augmented_q.shape)

    # 标准化处理
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(augmented_settings)
    y_scaled = scaler_y.fit_transform(augmented_q)

    # 标准化测试数据 (使用训练数据的scaler)
    X_test_scaled = scaler_X.transform(X_test_external)
    y_test_scaled = scaler_y.transform(y_test_external)  

    # 数据集划分
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_scaled, y_scaled, test_size=0.1, random_state=42
    # )
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42    # 20%验证，80%训练
    )

    # 数据加载器
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=128
    )

    # 模型初始化
    model = LipschitzNN()
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    
    # 优化配置
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.MSELoss()

    # 训练循环
    best_loss = float('inf')
    train_losses, val_losses = [], []
    lr_history = []
    
    print("\n=== 开始离线训练 ===")
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch.requires_grad_(True)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch) 
            # print(f'Loss MSE: {criterion(pred, y_batch)}')
            # print(f'Lipschitz Loss: {model.lip_lambda * model.lip_loss(X_batch, pred)}')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
        # 记录学习率
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        
        # 保存最佳模型
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: 保存最佳模型 (Val Loss: {best_loss:.4f})")
        
        # 记录损失
        train_losses.append(epoch_loss/len(train_loader))
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epoch} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 可视化训练过程
    plot_loss_curves(train_losses, val_losses)
    
    # 学习率变化曲线
    plt.figure(figsize=(10,4))
    plt.plot(lr_history)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()

    # 测试集评估
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X_test_scaled))
        y_pred = model.scaler_y.inverse_transform(y_pred_scaled.numpy())
        y_test_unscaled = model.scaler_y.inverse_transform(y_test_scaled)
        
        # 计算指标
        rmse = np.sqrt(np.mean((y_pred - y_test_unscaled)**2))
        r2 = r2_score(y_test_unscaled, y_pred)
        print(f"\n测试集综合指标:")
        print(f"- RMSE: {rmse:.3f} dB")
        print(f"- R²分数: {r2:.3f}")
        
        # 各维度指标
        print("\n各维度表现:")
        for i in range(y_pred.shape[1]):
            col_rmse = np.sqrt(np.mean((y_pred[:,i] - y_test_unscaled[:,i])**2))
            col_r2 = r2_score(y_test_unscaled[:,i], y_pred[:,i])
            print(f"Q{i+1}: RMSE={col_rmse:.3f} dB | R²={col_r2:.3f}")
        
        # 预测可视化
        plot_predictions(y_test_unscaled, y_pred)
    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    torch.save(checkpoint, save_path)
    
    return model



if __name__ == "__main__":
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    set_seed()
    # Lip_data = np.load('./Lip_GN_5wave.npz')
    # Lip = Lip_data['mgn_list'][-1]

    Lip_data = np.load('./Lip_GN_5wave_0319.npz')
    Lip = Lip_data['mgn_list'][-1]

    offline_training(num_epoch=300)
