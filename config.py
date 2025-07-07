import os
# -*- coding: utf-8 -*-
"""
统一配置文件
集中管理所有路径配置，便于项目迁移和部署
"""

import os
from pathlib import Path

# 项目根目录（自动检测）
PROJECT_ROOT = Path(__file__).parent.absolute()

# 基础路径配置
class PathConfig:
    """路径配置类，便于统一管理"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        
        # 数据相关路径
        self.data_dir = self.project_root.parent.parent / "data"
        self.temp_data_dir = self.project_root / "temp_data"
        
        # 模型相关路径
        self.models_dir = self.project_root / "models"
        self.emotion2vec_model_dir = self.models_dir / "iic" / "BaseModel_Emotion2Vec" / "best_model"
        self.emotion2vec_base_dir = self.models_dir / "iic" / "emotion2vec_base"
        
        # 配置文件路径
        self.configs_dir = self.project_root / "configs"
        self.base_model_config = self.configs_dir / "base_model.yml"
        
        # 数据集相关路径
        self.dataset_dir = self.project_root / "dataset"
        self.features_dir = self.dataset_dir / "features1"
        self.emotion2vec_features_dir = self.dataset_dir / "features_emotion2vec"
        self.custom_features_dir = self.dataset_dir / "features_custom"
        self.label_list_path = self.dataset_dir / "label_list.txt"
        self.standard_scaler_path = self.dataset_dir / "standard.m"
        
        # 结果输出路径
        self.results_dir = self.project_root / "results"
        
        # 模板文件路径
        self.template_file = self.project_root / "心理评估报告模板.json"
        
        # 向后兼容的路径变量
        self.PROJECT_PATH = self.project_root
        self.DATA_PATH = self.project_root / "data"
        self.DATA1_PATH = Path("E:/200_main_business/朔黄工程/第二阶段数据")
        self.DATASET_PATH = self.dataset_dir
        self.FEATURES_PATH = self.features_dir
        self.MODEL_PATH = self.models_dir
        self.EMOTION2VEC_MODEL_PATH = self.emotion2vec_base_dir
        self.EMOTION2VEC_FEATURES_PATH = self.emotion2vec_features_dir
        self.CUSTOM_FEATURES_PATH = self.custom_features_dir
        self.RESULTS_PATH = self.results_dir

    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.temp_data_dir,
            self.results_dir,
            self.models_dir,
            self.dataset_dir,
            self.features_dir,
            self.emotion2vec_features_dir,
            self.custom_features_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"确保目录存在: {directory}")
    
    def get_model_path(self, model_name="model.pth"):
        """获取模型文件路径"""
        return self.emotion2vec_model_dir / model_name
    
    def get_temp_user_dir(self, user_id):
        """获取用户临时数据目录"""
        user_dir = self.temp_data_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def validate_paths(self):
        """验证关键路径是否存在"""
        critical_paths = {
            "模型目录": self.emotion2vec_model_dir,
            "配置文件": self.base_model_config,
            "标签列表": self.label_list_path,
            "标准化文件": self.standard_scaler_path
        }
        
        missing_paths = []
        for name, path in critical_paths.items():
            if not path.exists():
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            print("警告：以下关键文件/目录不存在：")
            for missing in missing_paths:
                print(f"  - {missing}")
            return False
        
        print("所有关键路径验证通过")
        return True
    
    def to_dict(self):
        """返回路径配置字典"""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "temp_data_dir": str(self.temp_data_dir),
            "models_dir": str(self.models_dir),
            "emotion2vec_model_dir": str(self.emotion2vec_model_dir),
            "configs_dir": str(self.configs_dir),
            "dataset_dir": str(self.dataset_dir),
            "results_dir": str(self.results_dir),
            "template_file": str(self.template_file)
        }

# 创建全局配置实例
path_config = PathConfig()

# 向后兼容的路径变量（保持原有代码可用）
PROJECT_PATH = path_config.PROJECT_PATH
DATA_PATH = path_config.DATA_PATH
DATA1_PATH = path_config.DATA1_PATH
DATASET_PATH = path_config.DATASET_PATH
FEATURES_PATH = path_config.FEATURES_PATH
MODEL_PATH = path_config.MODEL_PATH
EMOTION2VEC_MODEL_PATH = path_config.EMOTION2VEC_MODEL_PATH
EMOTION2VEC_FEATURES_PATH = path_config.EMOTION2VEC_FEATURES_PATH
CUSTOM_FEATURES_PATH = path_config.CUSTOM_FEATURES_PATH
RESULTS_PATH = path_config.RESULTS_PATH

def get_paths():
    """获取所有路径配置（向后兼容）"""
    return path_config.to_dict()

def create_directories():
    """创建必要目录（向后兼容）"""
    path_config.create_directories()

if __name__ == "__main__":
    print("=" * 60)
    print("统一路径配置")
    print("=" * 60)
    
    # 创建目录
    create_directories()
    
    # 验证路径
    path_config.validate_paths()
    
    # 显示配置
    print("\n当前路径配置:")
    for key, value in get_paths().items():
        print(f"  {key}: {value}")
    
    print("\n配置完成！")
