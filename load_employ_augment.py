import torch
import numpy as np
import os
from Conditional_VAE_with_GAN_V1 import ConfigManager, AdvancedVAEcGAN, DataPreprocessor

def load_model_and_preprocessor(config_path, checkpoint_path):
    """加载训练好的模型和数据预处理器"""
    # 初始化配置
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 创建模型实例
    model = AdvancedVAEcGAN(config)
    
    # 加载训练好的权重
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    # 过滤掉不匹配的键
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    
    # 加载数据预处理器（需要复用训练时的统计量）
    data_preprocessor = DataPreprocessor(config_path)
    _, _, _, _ = data_preprocessor.load_and_preprocess()  # 仅加载统计量
    
    return model, data_preprocessor

def generate_augmented_data(model, preprocessor, input_data_path, target_size=1000):
    """生成增强数据"""
    # 加载需要增强的原始数据
    real_data = np.load(input_data_path)
    X_real = torch.from_numpy(real_data['settings']).float()
    Y_real = torch.from_numpy(real_data['q_values']).float()
    
    # 使用与训练数据相同的均值和标准差进行标准化
    X_real_norm = (X_real - preprocessor.X_mean) / (preprocessor.X_std + 1e-8)
    Y_real_norm = (Y_real - preprocessor.Y_mean) / (preprocessor.Y_std + 1e-8)
    
    # 使用模型进行增强
    with torch.no_grad():
        x_aug_norm, y_aug_norm = model.adaptive_augmentation(
            X_real_norm, 
            Y_real_norm,
            target_size=target_size
        )
    
    # 逆标准化还原数据
    x_aug_orig, y_aug_orig = preprocessor.inverse_transform(x_aug_norm, y_aug_norm)
    
    return x_aug_orig.numpy(), y_aug_orig.numpy()

def save_augmented_data(x_aug, y_aug, output_path):
    """保存增强后的数据"""
    np.savez(
        output_path,
        settings=x_aug,
        q_values=y_aug
    )
    print(f"增强数据已保存至 {output_path}，共生成 {len(x_aug)} 个样本")

if __name__ == "__main__":
    # 配置路径
    CONFIG_DATA_PATH = './lhs_dataset_5wave_500.npz'  # 训练使用的数据路径
    MODEL_CHECKPOINT = './checkpoints_vi/best_model.pth'  # 训练好的模型路径
    INPUT_DATA_PATH = './lhs_dataset_5wave_500.npz'  # 需要增强的原始数据路径
    OUTPUT_PATH = './augmented_data_1000_V2.npz'  # 输出文件路径
    
    # 加载模型和预处理器
    model, preprocessor = load_model_and_preprocessor(CONFIG_DATA_PATH, MODEL_CHECKPOINT)
    
    # 生成1000个增强样本
    x_aug, y_aug = generate_augmented_data(
        model, 
        preprocessor,
        INPUT_DATA_PATH,
        target_size=1000
    )
    
    # 保存结果
    save_augmented_data(x_aug, y_aug, OUTPUT_PATH)