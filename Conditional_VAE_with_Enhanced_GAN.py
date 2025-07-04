import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader

# ======================
# 可配置参数
# ======================
config = {
    "input_dim": 12,
    "output_dim": 5,
    "latent_dim": 10,
    "noise_dim": 6,
    "feature_scales": [16, 32, 64],  # 多尺度特征维度
    "attention_heads": 4,
    "batch_size": 4,
    "vae_beta": 0.7,
    "gp_weight": 10,
    "lr": {
        "encoder": 2e-4,
        "generator": 1e-4,
        "discriminator": 4e-4,
        "controller": 5e-4
    }
}

# ======================
# 自定义损失函数接口
# ======================
class CustomLoss(nn.Module):
    def __init__(self, base_loss='mse', **kwargs):
        super().__init__()
        self.base_loss = base_loss
        self.config = kwargs
        
        # 基础损失函数
        if base_loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif base_loss == 'mae':
            self.loss_fn = nn.L1Loss()
        elif base_loss == 'huber':
            self.loss_fn = nn.HuberLoss()
        elif base_loss == 'bce':  # 新增支持二元交叉熵
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"不支持的损失函数类型: {base_loss}")
    def forward(self, predictions, targets, model=None, epoch=None):
        """用户可重写此方法实现自定义损失"""
        base_loss = self.loss_fn(predictions, targets)
        return base_loss
# ======================
# 高级模型组件
# ======================
class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征金字塔"""
    def __init__(self, input_dim, output_dim, feature_scales):
        super().__init__()
        self.input_branches = nn.ModuleList()
        self.output_branches = nn.ModuleList()
        
        # 输入特征提取分支
        for dim in feature_scales:
            seq = nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(dim),
                nn.Linear(dim, dim)
            )
            self.input_branches.append(seq)
        
        # 输出特征提取分支
        for dim in feature_scales:
            seq = nn.Sequential(
                nn.Linear(output_dim, dim//2),
                nn.ReLU(),
                nn.Linear(dim//2, dim)
            )
            self.output_branches.append(seq)
        
        # 先不初始化融合层，在第一次前向传播后确定输入维度
        self.fusion = None
    
    def forward(self, x, y):
        # 多尺度特征提取
        input_features = []
        for branch in self.input_branches:
            input_features.append(branch(x))
        
        output_features = []
        for branch in self.output_branches:
            output_features.append(branch(y))
        
        # 跨尺度特征融合
        fused = []
        for i, (in_feat, out_feat) in enumerate(zip(input_features, output_features)):
            if i == 0:
                fused.append(torch.cat([in_feat, out_feat], dim=1))
            else:
                scale_factor = 2 ** (i-1)
                resized_in = F.adaptive_avg_pool1d(in_feat.unsqueeze(1), scale_factor).squeeze(1)
                resized_out = F.adaptive_avg_pool1d(out_feat.unsqueeze(1), scale_factor).squeeze(1)
                fused.append(torch.cat([resized_in, resized_out], dim=1))
        
        # 最终融合
        concat_features = torch.cat(fused, dim=1)
        
        # 第一次前向传播时初始化融合层
        if self.fusion is None:
            input_dim = concat_features.size(1)
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        
        return self.fusion(concat_features)

class ConditionalVAE(nn.Module):
    """高级条件VAE"""
    def __init__(self, input_dim, output_dim, latent_dim, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(64 + output_dim, 48),  # 64来自特征提取器
            nn.LeakyReLU(0.2),
            SelfAttention(48),
            nn.Linear(48, 32),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        
        # 解码器
        # 手动定义解码器的每一层，方便传入条件信息
        self.decoder_pre = nn.Sequential(
            nn.Linear(latent_dim + output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 48),
            nn.ReLU()
        )
        self.physics_layer = PhysicsInformedLayer(48, output_dim)
        self.decoder_post = nn.Linear(48, input_dim)
    
    def encode(self, x, y):
        features = self.feature_extractor(x, y)
        conditioned = torch.cat([features, y], dim=1)
        
        # 添加数值稳定性层
        conditioned = torch.clamp(conditioned, -10, 10)  # 限制输入范围
        
        h = self.encoder(conditioned)
        h = torch.clamp(h, -10, 10)  # 限制激活值范围
        return self.fc_mu(h), self.fc_var(h)


    def reparameterize(self, mu, logvar):
        # 严格限制logvar范围
        logvar = torch.clamp(logvar, min=-5, max=2)  # 缩小范围
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    def decode(self, z, y):
        conditioned = torch.cat([z, y], dim=1)
        h = self.decoder_pre(conditioned)
        h = self.physics_layer(h, y)  # 传入条件信息 y
        return self.decoder_post(h)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

class AdvancedGenerator(nn.Module):
    """带自注意力和条件批归一化的生成器"""
    def __init__(self, noise_dim, cond_dim, latent_dim):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 32)
        
        self.block1 = ConditionalBlock(noise_dim, 32, cond_dim=32)
        self.attn1 = SelfAttention(32)
        self.block2 = ConditionalBlock(32, 64, cond_dim=32)
        self.attn2 = SelfAttention(64)
        self.block3 = ConditionalBlock(64, latent_dim, cond_dim=32)
        
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    """带自注意力和条件批归一化的生成器（关键修改）"""
    def forward(self, noise, c):
        c_proj = self.cond_proj(c)  # 确保c的批次维度与noise一致
        h = self.block1(noise, c_proj)
        h = self.attn1(h)
        h = self.block2(h, c_proj)
        h = self.attn2(h)
        h = self.block3(h, c_proj)
        return self.out(h)

class SNDiscriminator(nn.Module):
    """谱归一化判别器（修复数值稳定性）"""
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.cond_proj = nn.utils.spectral_norm(nn.Linear(cond_dim, 32))
        
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim + 32, 64)),
            nn.LeakyReLU(0.2),
            SelfAttention(64),
            nn.utils.spectral_norm(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2),
            # 移除最后的Sigmoid激活函数
            nn.utils.spectral_norm(nn.Linear(32, 1))  
        )
    
    def forward(self, z, c):
        c_proj = self.cond_proj(c)
        inputs = torch.cat([z, c_proj], dim=1)
        logits = self.main(inputs)  # 现在输出logits而不是概率
        return logits  # 返回原始分数
# ======================
# 自定义模块
# ======================
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, dim = x.size()
        # 计算 query、key 和 value 输出的特征维度
        query_dim = self.query.out_features
        key_dim = self.key.out_features
        value_dim = dim

        Q = self.query(x).view(batch_size, -1, query_dim)
        K = self.key(x).view(batch_size, -1, key_dim)
        V = self.value(x).view(batch_size, -1, value_dim)

        # 计算注意力分数
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (query_dim ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)

        # 计算注意力输出
        out = torch.bmm(attn, V).view(batch_size, dim)
        return self.gamma * out + x
    

class ConditionalBlock(nn.Module):
    """条件批归一化块（修正维度对齐）"""
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.cond_gamma = nn.Linear(cond_dim, out_dim)  # 输出维度与out_dim一致
        self.cond_beta = nn.Linear(cond_dim, out_dim)   # 输出维度与out_dim一致
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x, c):
        h = self.fc(x)  # 形状：[batch_size, out_dim]
        h = self.bn(h)
        gamma = self.cond_gamma(c)  # 形状：[batch_size, out_dim]
        beta = self.cond_beta(c)    # 形状：[batch_size, out_dim]
        return self.act(h * (1 + gamma) + beta)  # 维度完全匹配

class PhysicsInformedLayer(nn.Module):
    """物理约束层（彻底重构）"""
    def __init__(self, dim, cond_dim):
        super().__init__()
        # 使用更安全的约束参数
        self.constraint_fc = nn.Linear(cond_dim, dim)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # 初始化小值

    def forward(self, x, c):
        # 移除所有可能导致数值不稳定的操作
        constraint = self.constraint_fc(c)
        # 使用残差连接添加约束
        return x + self.gamma * constraint

class MetaAugmentationController(nn.Module):
    """元学习增强控制器"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 输出: [噪声水平, 插值强度, 生成数量]
        )
    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        params = torch.sigmoid(self.net(inputs))
        return {
            'noise_level': params[:, 0],
            'mix_ratio': params[:, 1],
            'gen_factor': params[:, 2] * 5 + 1  # 1-6倍生成
        }

# ======================
# 组合模型与训练框架
# ======================
class AdvancedVAEcGAN(nn.Module):
    def __init__(self, config, custom_vae_loss=None, custom_gan_loss=None):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(
            config['input_dim'], 
            config['output_dim'], 
            config['feature_scales']
        )
        
        # 核心模型
        self.cvae = ConditionalVAE(
            config['input_dim'], 
            config['output_dim'], 
            config['latent_dim'], 
            self.feature_extractor
        )
        self.generator = AdvancedGenerator(
            config['noise_dim'], 
            config['output_dim'], 
            config['latent_dim']
        )
        self.discriminator = SNDiscriminator(
            config['latent_dim'], 
            config['output_dim']
        )
        self.controller = MetaAugmentationController(
            config['input_dim'], 
            config['output_dim']
        )
        
        # 自定义损失函数
        self.custom_vae_loss = custom_vae_loss or CustomLoss()
        self.custom_gan_loss = custom_gan_loss or CustomLoss(base_loss='bce')
        
        # 优化器
        self.opt_encoder = Adam(self.cvae.parameters(), lr=config['lr']['encoder'])
        self.opt_generator = Adam(self.generator.parameters(), lr=config['lr']['generator'])
        self.opt_discriminator = Adam(self.discriminator.parameters(), lr=config['lr']['discriminator'])
        self.opt_controller = RMSprop(self.controller.parameters(), lr=config['lr']['controller'])
    
    def compute_gradient_penalty(self, real_samples, fake_samples, conditions):
        """更稳定的梯度惩罚计算"""
        alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # 更严格的梯度截断
        interpolates = torch.clamp(interpolates, -10, 10)
        
        d_interpolates = self.discriminator(interpolates, conditions)
        
        # 创建与d_interpolates相同形状的全1张量
        grad_outputs = torch.ones_like(d_interpolates, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        
        # 更安全的梯度范数计算
        gradient_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-10)
        gradient_penalty = torch.clamp((gradient_norm - 1) ** 2, 0, 1e5).mean()  # 限制最大值
        
        return gradient_penalty
        
    def train_step(self, x_real, y_real, epoch):
        # 1. 训练VAE编码器
        self.opt_encoder.zero_grad()
        recon_x, mu, logvar = self.cvae(x_real, y_real)
        
        # 自定义VAE损失
        recon_loss = self.custom_vae_loss(recon_x, x_real, model=self, epoch=epoch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + self.config['vae_beta'] * kld_loss
        
        vae_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=1.0)
        self.opt_encoder.step()
        
        # 2. 训练判别器
        self.opt_discriminator.zero_grad()
        
        # 获取真实隐向量
        with torch.no_grad():
            mu_real, _ = self.cvae.encode(x_real, y_real)
            z_real = self.cvae.reparameterize(mu_real, logvar)
        
        # 生成假样本
        noise = torch.randn(x_real.size(0), self.config['noise_dim'], device=x_real.device)
        z_fake = self.generator(noise, y_real).detach()
        
        # 判别器损失
        real_validity = self.discriminator(z_real, y_real)
        fake_validity = self.discriminator(z_fake.detach(), y_real)
        
        # 自定义GAN损失
        d_loss_real = self.custom_gan_loss(real_validity, torch.ones_like(real_validity))
        d_loss_fake = self.custom_gan_loss(fake_validity, torch.zeros_like(fake_validity))
        
        # 梯度惩罚
        gp = self.compute_gradient_penalty(z_real, z_fake, y_real)
        
        d_loss = d_loss_real + d_loss_fake + self.config['gp_weight'] * gp
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.opt_discriminator.step()
        
        # 3. 训练生成器
        self.opt_generator.zero_grad()
        validity = self.discriminator(z_fake, y_real)
        g_loss = self.custom_gan_loss(validity, torch.ones_like(validity))
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.opt_generator.step()
        
        # 4. 训练元控制器
        self.opt_controller.zero_grad()
        aug_params = self.controller(x_real, y_real)
        
        # 使用增强参数生成样本
        with torch.no_grad():
            noise = torch.randn(x_real.size(0), self.config['noise_dim'], device=x_real.device)
            noise *= aug_params['noise_level'].unsqueeze(1)
            z_fake_meta = self.generator(noise, y_real)
            x_fake = self.cvae.decode(z_fake_meta, y_real)
        
        # 计算增强效果
        recon_fake, _, _ = self.cvae(x_fake, y_real)
        controller_loss = F.mse_loss(recon_fake, x_real)
        controller_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
        self.opt_controller.step()
        
        return {
            'vae_loss': vae_loss.item(),
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'controller_loss': controller_loss.item()
        }
    
    def generate_samples(self, y_conditions, num_samples=10, strategy='optimal'):
        """高级样本生成策略（修正条件混合逻辑）"""
        self.eval()
        
        if strategy == 'optimal':
            y_base = y_conditions.mean(dim=0, keepdim=True)
            x_dummy = torch.zeros(1, self.config['input_dim'], device=y_conditions.device)
            aug_params = self.controller(x_dummy, y_base)
            
            num_samples = int(aug_params['gen_factor'].item() * num_samples)
            noise_level = aug_params['noise_level'].item()
            mix_ratio = aug_params['mix_ratio'].item()
        else:
            noise_level = 0.1
            mix_ratio = 0.5
        
        # 生成隐向量（批次维度为num_samples）
        noise = torch.randn(num_samples, self.config['noise_dim'], device=y_conditions.device)
        noise *= noise_level
        
        # 条件混合（关键修正：确保mixed_y的批次维度为num_samples）
        if len(y_conditions) < num_samples:
            # 重复原始条件数据以匹配num_samples
            y_conditions = y_conditions.repeat((num_samples // len(y_conditions)) + 1, 1)
        y_conditions = y_conditions[:num_samples]  # 截断到目标数量
        
        if mix_ratio > 0:
            idx = torch.randperm(len(y_conditions))[:num_samples]  # 确保索引数量匹配
            mixed_y = mix_ratio * y_conditions + (1 - mix_ratio) * y_conditions[idx]
        else:
            mixed_y = y_conditions
        
        z_fake = self.generator(noise, mixed_y)  # 此时noise和mixed_y批次维度一致
        
        with torch.no_grad():
            x_generated = self.cvae.decode(z_fake, mixed_y)
        
        return x_generated, mixed_y
    
    def adaptive_augmentation(self, x, y, target_size=100):
        original_size = len(x)
        # 需要生成的总量 = 目标量 - 原始量（如果原始量已足够则无需生成）
        remaining = max(target_size - original_size, 0)
        gen_samples = []
        
        while remaining > 0:
            params = self.controller(x, y)
            # 限制生成数量不超过剩余需求
            batch_size = min(int(params['gen_factor'].mean().item() * 10), remaining)
            
            x_new, y_new = self.generate_samples(y, num_samples=batch_size, strategy='controller')
            
            gen_samples.append((x_new, y_new))
            remaining -= batch_size
        
        # 只保留目标数量的样本（包含原始数据）
        x_aug = torch.cat([x] + [xs[:min(len(xs), target_size - original_size)] for xs, _ in gen_samples])
        y_aug = torch.cat([y] + [ys[:min(len(ys), target_size - original_size)] for _, ys in gen_samples])
        return x_aug[:target_size], y_aug[:target_size]

# ======================
# 自定义损失函数示例
# ======================
class CustomVAELoss(CustomLoss):
    """示例：自定义VAE损失函数"""
    def __init__(self, base_loss='mse', alpha=0.5, beta=0.1):
        super().__init__(base_loss)
        self.alpha = alpha  # 特征重要性权重
        self.beta = beta    # 相关性惩罚系数
    
    def forward(self, predictions, targets, model=None, epoch=None):
        # 添加数值稳定性保护
        predictions = torch.clamp(predictions, -10, 10)
        targets = torch.clamp(targets, -10, 10)

        # 基础重建损失
        base_loss = super().forward(predictions, targets)
        
        # 特征重要性加权
        feature_weights = torch.linspace(1, 1.5, predictions.size(1))
        weighted_loss = (feature_weights * (predictions - targets)**2).mean()
        
        # 相关性保持损失
        if model is not None:
            input_corr = torch.corrcoef(targets.T)
            pred_corr = torch.corrcoef(predictions.T)
            corr_loss = F.mse_loss(input_corr, pred_corr)
        else:
            corr_loss = 0
        
        # 组合损失
        total_loss = base_loss + self.alpha * weighted_loss + self.beta * corr_loss
        
        # 周期性KL增强
        if epoch and epoch % 10 == 0:
            total_loss += 0.3 * torch.mean(model.cvae.fc_var.weight**2)
        
        return total_loss

class CustomGANLoss(CustomLoss):
    """使用BCEWithLogitsLoss代替BCELoss"""
    def __init__(self, base_loss='bce', gamma=0.5):
        super().__init__(base_loss)
        self.gamma = gamma
        # 替换为更稳定的损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()  # 自动处理logits输入
    
    def forward(self, predictions, targets, model=None, epoch=None):
        # 直接计算logits损失
        adv_loss = self.loss_fn(predictions, targets)
        
        # 特征匹配损失（保持不变）
        if model and hasattr(model.discriminator, 'get_features'):
            real_features = model.discriminator.get_features(model.z_real, model.y_real)
            fake_features = model.discriminator.get_features(model.z_fake, model.y_real)
            fm_loss = F.mse_loss(real_features, fake_features)
        else:
            fm_loss = 0
        
        return adv_loss + self.gamma * fm_loss
    
# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 1. 准备数据
    dataset = np.load('./results/gn_prediction_results.npz')
    # X = torch.randn(20, 12).float()  # 12维输入
    # Y = torch.randn(20, 5).float()   # 5维输出
    X = dataset['settings']
    Y = dataset['predicted_qs']
    # 将NumPy数组转换为PyTorch Tensor
    X = torch.from_numpy(X).float()  # 转换为float32类型的Tensor
    Y = torch.from_numpy(Y).float()
    # 数据标准化
    X_mean, X_std = X.mean(dim=0), X.std(dim=0)  # 计算输入特征的均值和标准差
    Y_mean, Y_std = Y.mean(dim=0), Y.std(dim=0)  # 计算输出目标的均值和标准差
    
    # 在标准化前检查标准差
    print("输入特征标准差:", X_std)
    print("输出特征标准差:", Y_std)

    # 如果标准差太小，使用最小阈值
    X_std[X_std < 1e-8] = 1e-8
    Y_std[Y_std < 1e-8] = 1e-8

    # 标准化（避免除零，添加极小值）
    X = (X - X_mean) / (X_std + 1e-8)
    Y = (Y - Y_mean) / (Y_std + 1e-8)  

    print("输入标准化X:", X)
    print("输出标准化Y:", Y)
    # 2. 创建模型
    custom_vae_loss = CustomVAELoss(alpha=0.7, beta=0.2)
    custom_gan_loss = CustomGANLoss(gamma=0.6)
    
    model = AdvancedVAEcGAN(
        config, 
        custom_vae_loss=custom_vae_loss,
        custom_gan_loss=custom_gan_loss
    )
    
    # # 3. 训练模型
    # for epoch in range(1000):
    #     idx = torch.randperm(len(X))
    #     for i in range(0, len(X), config['batch_size']):
    #         batch_idx = idx[i:i+config['batch_size']]
    #         x_batch = X[batch_idx]
    #         y_batch = Y[batch_idx]
            
    #         losses = model.train_step(x_batch, y_batch, epoch)
        
    #     if epoch % 10 == 0:
    #         print(f"Epoch {epoch}: VAE Loss {losses['vae_loss']:.4f}, "
    #               f"D Loss {losses['d_loss']:.4f}, G Loss {losses['g_loss']:.4f}")
    # 添加权重初始化函数
# 添加权重初始化函数
    def weights_init(m):
        if isinstance(m, nn.Linear):
            # 使用更稳定的Kaiming初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    # 应用权重初始化
    model.apply(weights_init)

    # 训练参数
    best_loss = float('inf')
    nan_encountered = False
    patience_counter = 0
    max_patience = 5
    original_lrs = {
        'encoder': config['lr']['encoder'],
        'generator': config['lr']['generator'],
        'discriminator': config['lr']['discriminator'],
        'controller': config['lr']['controller']
    }

    # 梯度累积步数
    accumulation_steps = 4
    gradient_accumulation_counter = 0

    # 训练模型
    for epoch in range(1000):
        epoch_losses = {
            'vae_loss': 0.0,
            'd_loss': 0.0,
            'g_loss': 0.0,
            'controller_loss': 0.0
        }
        
        # 打乱数据索引
        idx = torch.randperm(len(X))
        
        # 使用数据加载器
        for i in range(0, len(X), config['batch_size']):
            batch_idx = idx[i:i+config['batch_size']]
            x_batch = X[batch_idx]
            y_batch = Y[batch_idx]
            
            # 执行训练步骤
            losses = model.train_step(x_batch, y_batch, epoch)
            
            # 累积损失
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            # 梯度累积
            gradient_accumulation_counter += 1
            if gradient_accumulation_counter % accumulation_steps == 0:
                # 应用梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.cvae.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.controller.parameters(), max_norm=1.0)
                
                # 更新优化器
                model.opt_encoder.step()
                model.opt_discriminator.step()
                model.opt_generator.step()
                model.opt_controller.step()
                
                # 清空梯度
                model.opt_encoder.zero_grad()
                model.opt_discriminator.zero_grad()
                model.opt_generator.zero_grad()
                model.opt_controller.zero_grad()
        
        # 计算平均损失
        num_batches = max(1, len(X) // config['batch_size'])
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # 打印训练信息
        if epoch % 1 == 0:  # 每个epoch都打印
            print(f"\nEpoch {epoch}:")
            print(f"  VAE Loss: {epoch_losses['vae_loss']:.6f}")
            print(f"  D Loss: {epoch_losses['d_loss']:.6f}")
            print(f"  G Loss: {epoch_losses['g_loss']:.6f}")
            print(f"  Controller Loss: {epoch_losses['controller_loss']:.6f}")
            
            # 监控中间值
            with torch.no_grad():
                # 使用固定的小批量进行调试
                debug_batch = min(config['batch_size'], len(X))
                x_debug = X[:debug_batch]
                y_debug = Y[:debug_batch]
                
                # 检查模型输出范围
                recon_x, mu, logvar = model.cvae(x_debug, y_debug)
                print(f"  VAE Mu: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
                print(f"  VAE LogVar: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}, mean={logvar.mean().item():.4f}")
                
                # 检查重参数化后的z
                z_real = model.cvae.reparameterize(mu, logvar)
                print(f"  Z Real: min={z_real.min().item():.4f}, max={z_real.max().item():.4f}, mean={z_real.mean().item():.4f}")
                
                # 检查判别器输出
                validity = model.discriminator(z_real, y_debug)
                print(f"  Discriminator Output: min={validity.min().item():.4f}, max={validity.max().item():.4f}, mean={validity.mean().item():.4f}")
                
                # 检查VAE重建
                print(f"  Recon X: min={recon_x.min().item():.4f}, max={recon_x.max().item():.4f}, mean={recon_x.mean().item():.4f}")
                
                # 检查生成器输出
                noise = torch.randn(debug_batch, config['noise_dim'], device=x_debug.device)
                z_fake = model.generator(noise, y_debug)
                print(f"  Z Fake: min={z_fake.min().item():.4f}, max={z_fake.max().item():.4f}, mean={z_fake.mean().item():.4f}")
        
        # 检查NaN/Inf
        nan_inf_detected = False
        for key in epoch_losses:
            loss_value = epoch_losses[key]
            if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
                print(f"!!! NaN/Inf detected in {key} at epoch {epoch} !!!")
                nan_inf_detected = True
                nan_encountered = True
        
        # 如果检测到NaN/Inf，降低学习率或停止训练
        if nan_inf_detected:
            print("!!! Reducing learning rates due to NaN/Inf !!!")
            for g in model.opt_encoder.param_groups:
                g['lr'] *= 0.5
            for g in model.opt_discriminator.param_groups:
                g['lr'] *= 0.5
            for g in model.opt_generator.param_groups:
                g['lr'] *= 0.5
            for g in model.opt_controller.param_groups:
                g['lr'] *= 0.5
                
            print(f"New LRs - Encoder: {model.opt_encoder.param_groups[0]['lr']:.2e}, "
                f"Discriminator: {model.opt_discriminator.param_groups[0]['lr']:.2e}, "
                f"Generator: {model.opt_generator.param_groups[0]['lr']:.2e}, "
                f"Controller: {model.opt_controller.param_groups[0]['lr']:.2e}")
            
            # 如果学习率已经非常小，停止训练
            if model.opt_encoder.param_groups[0]['lr'] < 1e-8:
                print("!!! Learning rate too low, stopping training !!!")
                break
        
        # 检查损失改善情况
        current_loss = epoch_losses['vae_loss'] + epoch_losses['d_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch} - no improvement for {max_patience} epochs")
                break
        
        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {
                    'encoder': model.opt_encoder.state_dict(),
                    'discriminator': model.opt_discriminator.state_dict(),
                    'generator': model.opt_generator.state_dict(),
                    'controller': model.opt_controller.state_dict()
                },
                'losses': epoch_losses
            }, f'checkpoint_epoch_{epoch}.pth')
    
    
    # 4. 生成增强样本
    x_aug, y_aug = model.adaptive_augmentation(X, Y, target_size=100)
    print(f"增强后数据集: 输入={x_aug.shape}, 输出={y_aug.shape}")
    
    # 5. 保存生成样本
    torch.save({'inputs': x_aug, 'outputs': y_aug}, 'augmented_100data.pth')

    # 对增强样本进行逆标准化
    x_aug_original = x_aug * X_std + X_mean  # 逆标准化公式：原始数据 = 标准化数据 * 标准差 + 均值
    y_aug_original = y_aug * Y_std + Y_mean

    # 6. 保存原始尺度的增强样本
    torch.save({
        'inputs_original': x_aug_original,  # 原始尺度的输入
        'outputs_original': y_aug_original,  # 原始尺度的输出
        'inputs_normalized': x_aug,          # 标准化后的输入（可选保留）
        'outputs_normalized': y_aug           # 标准化后的输出（可选保留）
    }, 'augmented_100data_original.pth')

    print(f"增强后数据集（原始尺度）: 输入={x_aug_original.shape}, 输出={y_aug_original.shape}")


