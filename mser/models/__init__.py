'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-28 22:42:48
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-06-28 23:50:02
FilePath: \朔黄工程\第二阶段数据\code\语音端到端demo-0\语音端到端demo-1-1\mser\models\__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import importlib
from loguru import logger
from .base_model import BaseModel

__all__ = ['build_model']


def build_model(input_size, configs):
    use_model = configs.model_conf.get('model', 'BiLSTM')
    model_args = configs.model_conf.get('model_args', {})
    mod = importlib.import_module(__name__)
    model = getattr(mod, use_model)(input_size=input_size, **model_args)
    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
