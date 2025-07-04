def load_lhs_dataset(file_path, verbose=True):
    """
    加载LHS数据集，并可选择打印全部内容
    Parameters:
    file_path: str, 数据集文件路径
    verbose: bool, 是否详细打印所有字段内容
    Returns:
    dataset: dict, 数据集字典
    """
    import numpy as np

    data = np.load(file_path, allow_pickle=True)
    dataset = {key: data[key] for key in data.files}

    print(f"已加载数据集: {file_path}")
    print(f"样本数量: {dataset['n_samples']}")
    print(f"采集时间: {dataset['timestamp']}")

    if 'total_time' in dataset:
        print(f"总采集时间: {dataset['total_time']:.1f}秒 ({dataset['total_time']/60:.1f}分钟)")
        print(f"平均每个样本: {dataset['avg_time_per_sample']:.2f}秒")

    if verbose:
        print("\n数据集所有字段和内容：")
        for key in dataset:
            value = dataset[key]
            print(f"\n字段: {key}")
            print(f"类型: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"形状: {value.shape}")
                if value.size > 20:
                    print(f"前10项: {value.flat[:10]}")
                else:
                    print(value)
            else:
                print(value)
            print("-" * 40)

    return dataset

# 用法示例
if __name__ == "__main__":
    dataset = load_lhs_dataset("./lhs_dataset_5wave_500.npz", verbose=True)
    print("Finished")