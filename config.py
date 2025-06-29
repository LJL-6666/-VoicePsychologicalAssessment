import os
from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(os.path.abspath(__file__))

# 基础路径 - 项目根目录（当前文件所在目录）
PROJECT_PATH = current_file.parent

# 数据路径
DATA_PATH = PROJECT_PATH / "data"  # 原始数据目录
DATA1_PATH = Path("E:/200_main_business/朔黄工程/第二阶段数据")  # 筛选后的数据目录（问题5、8、14）

# 数据集路径
DATASET_PATH = PROJECT_PATH / "dataset"  # 生成的数据集目录
FEATURES_PATH = DATASET_PATH / "features1"  # 特征保存目录

# 模型路径
MODEL_PATH = PROJECT_PATH / "models"  # 模型保存目录
EMOTION2VEC_MODEL_PATH = MODEL_PATH / "iic" / "emotion2vec_base"  # Emotion2Vec模型路径

# 特征提取方法对应的特征目录
EMOTION2VEC_FEATURES_PATH = DATASET_PATH / "features_emotion2vec"  # Emotion2Vec特征保存目录
CUSTOM_FEATURES_PATH = DATASET_PATH / "features_custom"  # CustomFeature特征保存目录

# 结果路径
RESULTS_PATH = PROJECT_PATH / "results"  # 结果保存目录

# 创建必要的目录
def create_directories():
    """创建必要的目录"""
    directories = [
        DATA_PATH,
        DATA1_PATH,
        DATASET_PATH,
        FEATURES_PATH,
        MODEL_PATH,
        RESULTS_PATH,
        EMOTION2VEC_FEATURES_PATH,
        CUSTOM_FEATURES_PATH,
        EMOTION2VEC_MODEL_PATH.parent  # 创建iic目录
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("已创建必要的目录")

# 获取路径字典
def get_paths():
    """获取所有路径的字典"""
    return {
        "PROJECT_PATH": PROJECT_PATH,
        "DATA_PATH": DATA_PATH,
        "DATA1_PATH": DATA1_PATH,
        "DATASET_PATH": DATASET_PATH,
        "FEATURES_PATH": FEATURES_PATH,
        "TRAIN_LIST_PATH": TRAIN_LIST_PATH,
        "TEST_LIST_PATH": TEST_LIST_PATH,
        "TRAIN_FEATURES_LIST_PATH": TRAIN_FEATURES_LIST_PATH,
        "TEST_FEATURES_LIST_PATH": TEST_FEATURES_LIST_PATH,
        "MODEL_PATH": MODEL_PATH,
        "EMOTION2VEC_MODEL_PATH": EMOTION2VEC_MODEL_PATH,
        "EMOTION2VEC_FEATURES_PATH": EMOTION2VEC_FEATURES_PATH,
        "CUSTOM_FEATURES_PATH": CUSTOM_FEATURES_PATH,
        "RESULTS_PATH": RESULTS_PATH
    }

if __name__ == "__main__":
    # 如果直接运行此文件，则创建必要的目录并打印路径信息
    create_directories()
    
    # 打印所有路径
    paths = get_paths()
    print("\n路径配置:")
    for key, path in paths.items():
        print(f"{key}: {path}")
