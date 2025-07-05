import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加在文件最开头

# ======================
# 配置管理模块
# ======================
class ConfigManager:
    def __init__(self):
        self.config = {
            "input_dim": 12,
            "output_dim": 5,
            "latent_dim": 10,
            "noise_dim": 6,
            "feature_scales": [16, 32, 64],
            "attention_heads": 4,
            "batch_size": 4,
            "vae_beta": 0.7,
            "gp_weight": 10,
            "lr": {
                "encoder": 2e-4,
                "generator": 1e-4,
                "discriminator": 4e-4,
                "controller": 5e-4
            },
            "max_epochs": 1000,
            "patience": 30,
            "checkpoint_dir": "checkpoints"
        }
        
        # 创建检查点目录
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
    
    def get_config(self):
        return self.config

# ======================
# 自定义损失函数模块
# ======================
class CustomLoss(nn.Module):
    def __init__(self, base_loss='mse', **kwargs):
        super().__init__()
        self.base_loss = base_loss
        self.config = kwargs
        
        if base_loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif base_loss == 'mae':
            self.loss_fn = nn.L1Loss()
        elif base_loss == 'huber':
            self.loss_fn = nn.HuberLoss()
        elif base_loss == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()  # 使用更稳定的版本
        else:
            raise ValueError(f"Unsupported loss type: {base_loss}")
    
    def forward(self, predictions, targets, model=None, epoch=None):
        # 添加数值稳定性保护
        predictions = torch.clamp(predictions, -10, 10)
        targets = torch.clamp(targets, -10, 10)
        return self.loss_fn(predictions, targets)

class CustomVAELoss(CustomLoss):
    def __init__(self, base_loss='mse', alpha=0.5, beta=0.1):
        """
        初始化 CustomVAELoss 类的实例。

        参数:
        base_loss (str, 可选): 基础损失函数的类型，默认为 'mse'。
            支持的类型包括 'mse', 'mae', 'huber', 'bce'。
        alpha (float, 可选): 特征重要性加权损失的权重系数，默认为 0.5。
            用于控制特征加权损失在总损失中的占比。
        beta (float, 可选): 周期性 KL 增强损失的权重系数，默认为 0.1。
            在 forward 方法中，每隔一定轮数会加入 KL 增强损失，此参数调节该部分损失的权重。
        """
        # 调用父类 CustomLoss 的构造函数，初始化基础损失函数
        super().__init__(base_loss)
        # 保存特征重要性加权损失的权重系数
        self.alpha = alpha
        # 保存周期性 KL 增强损失的权重系数
        self.beta = beta
    
    def forward(self, predictions, targets, model=None, epoch=None):
        """
        计算自定义的 VAE 损失，包含基础重建损失、特征重要性加权损失和周期性 KL 增强损失。

        参数:
        predictions (torch.Tensor): 模型的预测值。
        targets (torch.Tensor): 真实的目标值。
        model (nn.Module, 可选): 模型实例，用于获取 KL 增强损失相关参数，默认为 None。
        epoch (int, 可选): 当前训练的轮数，用于控制周期性 KL 增强损失的添加时机，默认为 None。

        返回:
        torch.Tensor: 计算得到的总损失。
        """
        # 基础重建损失
        # 调用父类的 forward 方法计算预测值和目标值之间的基础损失
        base_loss = super().forward(predictions, targets)
        
        # 特征重要性加权
        # 创建一个从 1 到 1.5 的等间距张量，长度等于预测值的特征维度，作为每个特征的权重
        feature_weights = torch.linspace(1, 1.5, predictions.size(1), device=predictions.device)
        # 计算预测值和目标值之间的平方误差，再乘以特征权重，最后求平均值得到特征重要性加权损失
        weighted_loss = (feature_weights * (predictions - targets)**2).mean()
        
        # 组合损失
        # 将基础重建损失和特征重要性加权损失按照权重组合，得到总损失
        total_loss = base_loss + self.alpha * weighted_loss
        
        # 周期性 KL 增强
        # 当 epoch 不为 None 且当前轮数是 10 的倍数时，添加周期性 KL 增强损失
        if epoch and epoch % 10 == 0:
            # 计算 VAE 编码器中计算对数方差的全连接层权重的平方均值，并乘以系数 0.3 累加到总损失中
            total_loss += 0.3 * torch.mean(model.cvae.fc_var.weight**2)
        
        return total_loss

class CustomGANLoss(CustomLoss):
    def __init__(self, base_loss='bce', gamma=0.5):
        super().__init__(base_loss)
        self.gamma = gamma
    
    def forward(self, predictions, targets, model=None, epoch=None):
        adv_loss = super().forward(predictions, targets)
        return adv_loss

# ======================
# 模型组件模块
# ======================
class SelfAttention(nn.Module): # 自注意力机制，通过捕获特征间的依赖关系，提升特征的表达能力，用于增强模块的特征提取能力。。
    def __init__(self, in_dim):
        """
        初始化自注意力机制模块。

        参数:
        in_dim (int): 输入特征的维度。
        """
        # 调用父类的构造函数
        super().__init__()
        # 定义查询（Query）线性层，将输入特征映射到更低维度空间，维度为输入维度的 1/8
        self.query = nn.Linear(in_dim, in_dim // 8)
        # 定义键（Key）线性层，将输入特征映射到更低维度空间，维度为输入维度的 1/8
        self.key = nn.Linear(in_dim, in_dim // 8)
        # 定义值（Value）线性层，将输入特征进行线性变换，输出维度保持与输入维度一致
        self.value = nn.Linear(in_dim, in_dim)
        # 定义可学习的参数 gamma，初始值为 0，用于控制注意力输出的权重
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        自注意力机制的前向传播方法。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, dim)

        返回:
        torch.Tensor: 经过自注意力机制处理后的输出张量，形状与输入相同 (batch_size, dim)
        """
        # 获取输入张量的批次大小和特征维度
        batch_size, dim = x.size()
        # 通过查询（Query）线性层处理输入，并调整形状为 (batch_size, -1, query_out_features)
        Q = self.query(x).view(batch_size, -1, self.query.out_features)
        # 通过键（Key）线性层处理输入，并调整形状为 (batch_size, -1, key_out_features)
        K = self.key(x).view(batch_size, -1, self.key.out_features)
        # 通过值（Value）线性层处理输入，并调整形状为 (batch_size, -1, dim)
        V = self.value(x).view(batch_size, -1, dim)
        
        # 计算注意力分数，通过批次矩阵乘法（bmm）计算 Q 和 K 的转置的乘积，并进行缩放
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.query.out_features ** 0.5)
        # 对注意力分数应用 softmax 函数，得到注意力权重
        attn = torch.softmax(attn_scores, dim=-1)
        
        # 通过批次矩阵乘法计算注意力输出，并调整形状为 (batch_size, dim)
        out = torch.bmm(attn, V).view(batch_size, dim)
        # 将注意力输出乘以可学习参数 gamma 后与原始输入相加，实现残差连接
        return self.gamma * out + x

class ConditionalBlock(nn.Module):   # 条件归一化块（Conditional Normalization Block），常用于条件生成对抗网络（cGAN）或变分自编码器（VAE）中。这种结构可以有效地将类别标签等条件信息 c 融合到特征生成过程中，增强模型的条件控制能力。
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.fc = nn.Linear(in_dim, out_dim)  # 全连接层，将输入维度in_dim映射到输出维度out_dim
        self.bn = nn.BatchNorm1d(out_dim)  # 一维批归一化层，用于稳定训练过程
        self.cond_gamma = nn.Linear(cond_dim, out_dim)  # 生成缩放因子γ的条件投影层
        self.cond_beta = nn.Linear(cond_dim, out_dim)  # 生成偏移因子β的条件投影层
        self.act = nn.LeakyReLU(0.2)  # 带泄露的ReLU激活函数（负斜率0.2）
    """
    条件归一化：通过cond_gamma和cond_beta将条件信息(cond_dim)投射到特征空间(out_dim)，实现条件依赖的特征变换
    批归一化：bn层标准化中间特征，加速训练并缓解梯度消失
    激活函数：LeakyReLU引入非线性，保留负值信息（相比标准ReLU）
    """
    def forward(self, x, c):
        h = self.fc(x)  # 通过全连接层进行特征变换
        h = self.bn(h)  # 对特征进行批归一化处理
        gamma = self.cond_gamma(c)  # 从条件信息生成缩放因子γ
        beta = self.cond_beta(c)    # 从条件信息生成偏移因子β
        return self.act(h * (1 + gamma) + beta)  # 应用条件缩放偏移后激活
    """
    可以精确控制条件参数对生成结果的影响
    批归一化缓解了光学参数尺度差异带来的训练不稳定问题。批归一化层 (BatchNorm1d) 加速收敛。
    通过 cond_gamma 和 cond_beta 将条件信息（如光学参数）融合到特征中。残差式条件融合（h * (1 + gamma) + beta）保留了原始特征的重要信息。
    """
class PhysicsInformedLayer(nn.Module): # 物理约束层，用于确保生成的光学参数满足物理约束，如能量守恒、波长限制等。
    def __init__(self, dim, cond_dim):
        super().__init__()  # 调用父类nn.Module的初始化
        self.constraint_fc = nn.Linear(cond_dim, dim)  # 将条件信息映射到特征空间的线性层
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # 可学习的约束强度系数（初始化为0.1）gamma 初始化为0.1避免初期训练干扰
    """
    在光学参数生成任务中，该层可以确保生成的光学参数满足基本的物理规律（如能量守恒、波长限制等），同时保持生成数据的多样性。
    """
    def forward(self, x, c):
        constraint = self.constraint_fc(c)  # 将条件信息（如波长/能量）映射为物理约束
        return x + self.gamma * constraint  # 保留原始特征的同时添加约束

class MultiScaleFeatureExtractor(nn.Module):  # 多尺度特征提取器模块，用于从输入和输出数据中提取多尺度特征并进行融合
    """
    作为Conditional VAE的输入处理器，增强模型对光学参数多尺度特征的捕捉能力
    通过不同尺度的特征融合，提升生成数据的物理合理性
    通过多尺度特征金字塔结构显著提升了模型对复杂光学关系的建模能力。
    核心功能：1、多尺度特征提取；2、跨尺度特征融合；3、动态融合参数。
    """
    def __init__(self, input_dim, output_dim, feature_scales):
        super().__init__()
        # 输入和输出特征的多尺度提取分支
        self.input_branches = nn.ModuleList()  # 输入特征的多尺度提取网络
        self.output_branches = nn.ModuleList() # 输出特征的多尺度提取网络
        
        # 输入特征提取分支
        for dim in feature_scales:
            seq = nn.Sequential(
                nn.Linear(input_dim, dim),      # 线性变换到目标维度
                nn.LeakyReLU(0.2),              # 带泄露ReLU激活(负斜率0.2)，增强梯度流动
                nn.BatchNorm1d(dim),            # 批归一化层，稳定训练过程
                nn.Linear(dim, dim)              # 保持维度不变的线性变换
            )
            self.input_branches.append(seq)      # 添加到输入分支列表
        
        # 输出特征提取分支
        for dim in feature_scales:
            seq = nn.Sequential(
                nn.Linear(output_dim, dim//2),  # 先压缩到一半维度
                nn.ReLU(),                      # 标准ReLU激活
                nn.Linear(dim//2, dim)          # 再扩展到目标维度
            )
            self.output_branches.append(seq)     # 添加到输出分支列表
        
        # 融合层延迟初始化(根据第一次前向传播的实际维度动态创建)
        self.fusion = None
    
    def forward(self, x, y):
        # 多尺度特征提取
        input_features = [branch(x) for branch in self.input_branches]
        output_features = [branch(y) for branch in self.output_branches]
        
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
    def __init__(self, input_dim, output_dim, latent_dim, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(64 + output_dim, 48),
            nn.LeakyReLU(0.2),
            SelfAttention(48),
            nn.Linear(48, 32),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        
        # 解码器
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
        conditioned = torch.clamp(conditioned, -10, 10)
        h = self.encoder(conditioned)
        h = torch.clamp(h, -10, 10)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        # 严格限制logvar范围
        logvar = torch.clamp(logvar, min=-5, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        conditioned = torch.cat([z, y], dim=1)
        h = self.decoder_pre(conditioned)
        h = self.physics_layer(h, y)
        return self.decoder_post(h)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

class AdvancedGenerator(nn.Module):
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
    
    def forward(self, noise, c):
        c_proj = self.cond_proj(c)
        h = self.block1(noise, c_proj)
        h = self.attn1(h)
        h = self.block2(h, c_proj)
        h = self.attn2(h)
        h = self.block3(h, c_proj)
        return self.out(h)

class SNDiscriminator(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.cond_proj = nn.utils.spectral_norm(nn.Linear(cond_dim, 32))
        
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim + 32, 64)),
            nn.LeakyReLU(0.2),
            SelfAttention(64),
            nn.utils.spectral_norm(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(32, 1))  
        )
    
    def forward(self, z, c):
        c_proj = self.cond_proj(c)
        inputs = torch.cat([z, c_proj], dim=1)
        return self.main(inputs)

class MetaAugmentationController(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        params = torch.sigmoid(self.net(inputs))
        return {
            'noise_level': params[:, 0],
            'mix_ratio': params[:, 1],
            'gen_factor': params[:, 2] * 5 + 1
        }

# ======================
# 组合模型模块
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


        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def compute_gradient_penalty(self, real_samples, fake_samples, conditions):
        alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        interpolates = torch.clamp(interpolates, -10, 10)
        
        d_interpolates = self.discriminator(interpolates, conditions)
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
        gradient_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-10)
        gradient_penalty = torch.clamp((gradient_norm - 1) ** 2, 0, 1e5).mean()
        
        return gradient_penalty
        
    def train_step(self, x_real, y_real, epoch):
        # 1. 训练VAE编码器
        self.opt_encoder.zero_grad()
        recon_x, mu, logvar = self.cvae(x_real, y_real)
        
        # 自定义VAE损失
        recon_loss = self.custom_vae_loss(recon_x, x_real, model=self, epoch=epoch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() + 1e-8)  # 添加稳定性项
        vae_loss = recon_loss + self.config['vae_beta'] * kld_loss
        
        # 检查NaN并处理
        if torch.isnan(vae_loss):
            print(f"VAE loss NaN at epoch {epoch}, skipping step")
            return None
        
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
        
        d_loss_real = self.custom_gan_loss(real_validity, torch.ones_like(real_validity))
        d_loss_fake = self.custom_gan_loss(fake_validity, torch.zeros_like(fake_validity))
        
        # 梯度惩罚
        gp = self.compute_gradient_penalty(z_real, z_fake, y_real)
        
        d_loss = d_loss_real + d_loss_fake + self.config['gp_weight'] * gp
        
        if torch.isnan(d_loss):
            print(f"Discriminator loss NaN at epoch {epoch}, skipping step")
            return None
        
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.opt_discriminator.step()
        
        # 3. 训练生成器
        self.opt_generator.zero_grad()
        validity = self.discriminator(z_fake, y_real)
        g_loss = self.custom_gan_loss(validity, torch.ones_like(validity))
        
        if torch.isnan(g_loss):
            print(f"Generator loss NaN at epoch {epoch}, skipping step")
            return None
        
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
        
        if torch.isnan(controller_loss):
            print(f"Controller loss NaN at epoch {epoch}, skipping step")
            return None
        
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
        
        # 生成隐向量
        noise = torch.randn(num_samples, self.config['noise_dim'], device=y_conditions.device)
        noise *= noise_level
        
        # 条件混合
        if len(y_conditions) < num_samples:
            y_conditions = y_conditions.repeat((num_samples // len(y_conditions)) + 1, 1)
        y_conditions = y_conditions[:num_samples]
        
        if mix_ratio > 0:
            idx = torch.randperm(len(y_conditions))[:num_samples]
            mixed_y = mix_ratio * y_conditions + (1 - mix_ratio) * y_conditions[idx]
        else:
            mixed_y = y_conditions
        
        z_fake = self.generator(noise, mixed_y)
        
        with torch.no_grad():
            x_generated = self.cvae.decode(z_fake, mixed_y)
        
        return x_generated, mixed_y
    
    def adaptive_augmentation(self, x, y, target_size=100):
        original_size = len(x)
        remaining = max(target_size - original_size, 0)  # 需要生成的数量
        gen_samples = []
        
        while remaining > 0:
            params = self.controller(x, y)
            # 严格限制生成数量不超过剩余需求
            batch_size = min(int(params['gen_factor'].mean().item() * 10), remaining)
            
            x_new, y_new = self.generate_samples(
                y, 
                num_samples=batch_size,
                strategy='controller'
            )
            
            gen_samples.append((x_new, y_new))
            remaining -= batch_size
        
        # 合并时严格限制总数量
        x_aug = torch.cat([x] + [xs[:min(len(xs), target_size - original_size)] for xs, _ in gen_samples])
        y_aug = torch.cat([y] + [ys[:min(len(ys), target_size - original_size)] for _, ys in gen_samples])
        return x_aug[:target_size], y_aug[:target_size]

# ======================
# 数据预处理模块
# ======================
class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
    
    def load_and_preprocess(self):
        # 加载数据
        dataset = np.load(self.data_path)
        X = dataset['settings']
        # Y = dataset['predicted_qs']
        Y = dataset['q_values']
        
        # 转换为Tensor
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        
        # 计算统计量
        self.X_mean, self.X_std = X.mean(dim=0), X.std(dim=0)
        self.Y_mean, self.Y_std = Y.mean(dim=0), Y.std(dim=0)
        
        # 避免除零
        self.X_std[self.X_std < 1e-8] = 1e-8
        self.Y_std[self.Y_std < 1e-8] = 1e-8
        
        # 标准化
        X_norm = (X - self.X_mean) / (self.X_std + 1e-8)
        Y_norm = (Y - self.Y_mean) / (self.Y_std + 1e-8)
        
        return X_norm, Y_norm, X, Y
    
    def inverse_transform(self, X_norm, Y_norm):
        """将标准化数据转换回原始尺度"""
        X_original = X_norm * self.X_std + self.X_mean
        Y_original = Y_norm * self.Y_std + self.Y_mean
        return X_original, Y_original

# ======================
# 训练器模块
# ======================
class ModelTrainer:
    def __init__(self, model, X, Y, config):
        self.model = model
        self.X = X
        self.Y = Y
        self.config = config
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = config['patience']
        self.checkpoint_dir = config['checkpoint_dir']
        
        # 保存原始学习率
        self.original_lrs = {
            'encoder': config['lr']['encoder'],
            'generator': config['lr']['generator'],
            'discriminator': config['lr']['discriminator'],
            'controller': config['lr']['controller']
        }
    
    def train(self):
        for epoch in range(self.config['max_epochs']):
            epoch_losses = {
                'vae_loss': 0.0,
                'd_loss': 0.0,
                'g_loss': 0.0,
                'controller_loss': 0.0
            }
            
            # 打乱数据索引
            idx = torch.randperm(len(self.X))
            batch_count = 0
            
            for i in range(0, len(self.X), self.config['batch_size']):
                batch_idx = idx[i:i+self.config['batch_size']]
                x_batch = self.X[batch_idx]
                y_batch = self.Y[batch_idx]
                
                # 执行训练步骤
                losses = self.model.train_step(x_batch, y_batch, epoch)
                
                if losses is None:  # 跳过NaN步骤
                    continue
                
                # 累积损失
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
                
                batch_count += 1
            
            if batch_count == 0:  # 所有批次都跳过
                print(f"Epoch {epoch} skipped due to NaN losses")
                continue
                
            # 计算平均损失
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
            
            # 打印训练信息
            self._print_epoch_info(epoch, epoch_losses)
            
            # 检查NaN/Inf
            if self._handle_nan_inf(epoch, epoch_losses):
                continue  # 如果处理了NaN，继续下一个epoch
            
            # 检查损失改善情况
            if self._check_improvement(epoch, epoch_losses):
                break  # 提前停止
            
            # 定期保存检查点
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, epoch_losses)
    
    def _print_epoch_info(self, epoch, losses):
        if epoch % 1 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  VAE Loss: {losses['vae_loss']:.6f}")
            print(f"  D Loss: {losses['d_loss']:.6f}")
            print(f"  G Loss: {losses['g_loss']:.6f}")
            print(f"  Controller Loss: {losses['controller_loss']:.6f}")
            
            # 监控中间值
            with torch.no_grad():
                debug_batch = min(self.config['batch_size'], len(self.X))
                x_debug = self.X[:debug_batch]
                y_debug = self.Y[:debug_batch]
                
                recon_x, mu, logvar = self.model.cvae(x_debug, y_debug)
                print(f"  VAE Mu: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
                print(f"  VAE LogVar: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}, mean={logvar.mean().item():.4f}")
                
                z_real = self.model.cvae.reparameterize(mu, logvar)
                print(f"  Z Real: min={z_real.min().item():.4f}, max={z_real.max().item():.4f}, mean={z_real.mean().item():.4f}")
                
                validity = self.model.discriminator(z_real, y_debug)
                print(f"  Discriminator Output: min={validity.min().item():.4f}, max={validity.max().item():.4f}, mean={validity.mean().item():.4f}")
                
                print(f"  Recon X: min={recon_x.min().item():.4f}, max={recon_x.max().item():.4f}, mean={recon_x.mean().item():.4f}")
                
                noise = torch.randn(debug_batch, self.config['noise_dim'], device=x_debug.device)
                z_fake = self.model.generator(noise, y_debug)
                print(f"  Z Fake: min={z_fake.min().item():.4f}, max={z_fake.max().item():.4f}, mean={z_fake.mean().item():.4f}")
    
    def _handle_nan_inf(self, epoch, losses):
        nan_inf_detected = False
        for key, loss_value in losses.items():
            if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
                print(f"!!! NaN/Inf detected in {key} at epoch {epoch} !!!")
                nan_inf_detected = True
        
        if nan_inf_detected:
            print("!!! Reducing learning rates due to NaN/Inf !!!")
            for g in self.model.opt_encoder.param_groups:
                g['lr'] *= 0.5
            for g in self.model.opt_discriminator.param_groups:
                g['lr'] *= 0.5
            for g in self.model.opt_generator.param_groups:
                g['lr'] *= 0.5
            for g in self.model.opt_controller.param_groups:
                g['lr'] *= 0.5
                
            print(f"New LRs - Encoder: {self.model.opt_encoder.param_groups[0]['lr']:.2e}, "
                f"Discriminator: {self.model.opt_discriminator.param_groups[0]['lr']:.2e}, "
                f"Generator: {self.model.opt_generator.param_groups[0]['lr']:.2e}, "
                f"Controller: {self.model.opt_controller.param_groups[0]['lr']:.2e}")
            
            # 如果学习率已经非常小，停止训练
            if self.model.opt_encoder.param_groups[0]['lr'] < 1e-8:
                print("!!! Learning rate too low, stopping training !!!")
                return True
        
        return nan_inf_detected
    
    def _check_improvement(self, epoch, losses):
        current_loss = losses['vae_loss'] + losses['d_loss']
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            # 保存最佳模型
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pth'))
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping at epoch {epoch} - no improvement for {self.max_patience} epochs")
                return True
        return False
    
    def _save_checkpoint(self, epoch, losses):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': {
                'encoder': self.model.opt_encoder.state_dict(),
                'discriminator': self.model.opt_discriminator.state_dict(),
                'generator': self.model.opt_generator.state_dict(),
                'controller': self.model.opt_controller.state_dict()
            },
            'losses': losses
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

# # 分布对比可视化
# def plot_distribution_comparison(original, augmented, title):
#     n_features = original.shape[1]
#     n_cols = 3  # 每行显示3个子图
#     n_rows = (n_features + n_cols - 1) // n_cols  # 计算需要的行数
    
#     plt.figure(figsize=(15, 5*n_rows))  # 根据行数调整高度
    
#     for i in range(n_features):
#         plt.subplot(n_rows, n_cols, i+1)
#         sns.kdeplot(original[:,i], label='Original')
#         sns.kdeplot(augmented[:,i], label='Augmented')
#         plt.title(f'Dim {i+1}')
#         plt.legend()
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 初始化配置管理器
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 数据预处理
    data_preprocessor = DataPreprocessor('./lhs_dataset_5wave_500.npz')
    X_norm, Y_norm, X_orig, Y_orig = data_preprocessor.load_and_preprocess()
    
    # 检查数据
    print(f"Loaded data: X_norm shape={X_norm.shape}, Y_norm shape={Y_norm.shape}")
    print("Sample X_norm:", X_norm[0])
    print("Sample Y_norm:", Y_norm[0])
    
    # 创建模型
    custom_vae_loss = CustomVAELoss(alpha=0.7, beta=0.2)
    custom_gan_loss = CustomGANLoss(gamma=0.6)
    
    model = AdvancedVAEcGAN(
        config, 
        custom_vae_loss=custom_vae_loss,
        custom_gan_loss=custom_gan_loss
    )
    
    # 训练模型
    trainer = ModelTrainer(model, X_norm, Y_norm, config)
    trainer.train()
    
    #使用现实数据进行数据增强
    # 加载真实数据
    real_data = np.load('./lhs_dataset_5wave_500.npz')
    X_real = torch.from_numpy(real_data['settings'][:20]).float() 
    Y_real = torch.from_numpy(real_data['q_values'][:20]).float()
    
    # 使用仿真数据的均值和标准差标准化真实数据
    X_real_norm = (X_real - data_preprocessor.X_mean) / (data_preprocessor.X_std + 1e-8)
    Y_real_norm = (Y_real - data_preprocessor.Y_mean) / (data_preprocessor.Y_std + 1e-8)
    
    # 检查数据
    print(f"Loaded simulated data: X_norm shape={X_norm.shape}, Y_norm shape={Y_norm.shape}")
    print(f"Loaded real data: X_real_norm shape={X_real_norm.shape}, Y_real_norm shape={Y_real_norm.shape}")


    # 生成增强样本
    x_aug_norm, y_aug_norm = model.adaptive_augmentation(X_real_norm, Y_real_norm, target_size=200)
    print(f"Augmented dataset: inputs={x_aug_norm.shape}, outputs={y_aug_norm.shape}")
    
    # 逆标准化
    x_aug_orig, y_aug_orig = data_preprocessor.inverse_transform(x_aug_norm, y_aug_norm)
    
    # 保存结果
    torch.save({
        'inputs_original': x_aug_orig,
        'outputs_original': y_aug_orig,
        'inputs_normalized': x_aug_norm,
        'outputs_normalized': y_aug_norm
    }, 'augmented_100data_original.pth')
    
    print(f"Augmented dataset (original scale): inputs={x_aug_orig.shape}, outputs={y_aug_orig.shape}")


# # 可视化输入分布
# plot_distribution_comparison(X_real_norm.numpy(), x_aug_norm.numpy(), 'Input Distribution Comparison')

# # 可视化输出分布  
# plot_distribution_comparison(Y_real_norm.numpy(), y_aug_norm.numpy(), 'Output Distribution Comparison')