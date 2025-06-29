# -*- coding: utf-8 -*-
import sys
import os
import argparse
import json
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path

# 设置标准输出编码为UTF-8，避免Windows终端乱码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(current_dir)

# 导入配置文件中的路径
try:
    from config import RESULTS_PATH, DATA1_PATH, EMOTION2VEC_MODEL_PATH, MODEL_PATH
    config_available = True
except ImportError:
    config_available = False
    logging.warning("无法导入config模块，将使用默认路径")

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 情绪标签映射 (中文)
EMOTION_LABELS = [
    "中性", "快乐", "恐惧", "悲伤", "惊讶", "愤怒", "平静", "厌恶"
]

# 检查必要的依赖库
DEPENDENCIES_AVAILABLE = True
MISSING_DEPENDENCIES = []

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCIES.append(f"torch: {str(e)}")

try:
    import torchaudio
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCIES.append(f"torchaudio: {str(e)}")

try:
    from mser.data_utils.featurizer import AudioFeaturizer
    from mser.predict import MSERPredictor
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCIES.append(f"AudioFeaturizer/MSERPredictor: {str(e)}")

if DEPENDENCIES_AVAILABLE:
    class BaseModel(torch.nn.Module):
        def __init__(self, input_size=768, num_class=len(EMOTION_LABELS), hidden_size=256):
            super().__init__()
            self.pre_net = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3)
            )
            self.emotion_net = torch.nn.Linear(hidden_size, num_class)

        def forward(self, x):
            shared_features = self.pre_net(x)
            emotion_output = self.emotion_net(shared_features)
            return emotion_output

    def process_audio(waveform, sample_rate):
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform

    def is_silent(waveform, threshold=0.01, min_silence_duration=0.8):
        """
        检测音频是否为静音
        
        参数:
        waveform: torch.Tensor, 音频波形数据
        threshold: float, 静音阈值（相对于最大振幅）
        min_silence_duration: float, 最小静音持续时间比例（0-1）
        
        返回:
        bool: True表示静音，False表示非静音
        """
        try:
            # 计算音频的RMS能量
            rms = torch.sqrt(torch.mean(waveform ** 2))
            
            # 计算最大振幅
            max_amplitude = torch.max(torch.abs(waveform))
            
            # 如果最大振幅过小，认为是静音
            if max_amplitude < threshold:
                return True
            
            # 计算静音帧的比例
            frame_size = int(0.025 * 16000)  # 25ms帧
            hop_size = int(0.01 * 16000)     # 10ms跳跃
            
            silent_frames = 0
            total_frames = 0
            
            for i in range(0, len(waveform) - frame_size, hop_size):
                frame = waveform[i:i + frame_size]
                frame_rms = torch.sqrt(torch.mean(frame ** 2))
                
                if frame_rms < threshold:
                    silent_frames += 1
                total_frames += 1
            
            if total_frames == 0:
                return True
            
            silence_ratio = silent_frames / total_frames
            
            # 如果静音帧比例超过阈值，认为是静音
            return silence_ratio > min_silence_duration
            
        except Exception as e:
            logger.warning(f"静音检测时出错: {str(e)}，默认为非静音")
            return False

    def extract_features(audio_path, featurizer):
        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = process_audio(waveform, sr)
            
            # 静音检测
            if is_silent(waveform[0]):
                logger.info(f"检测到静音文件: {audio_path}")
                return "SILENT"  # 返回特殊标记表示静音
            
            waveform_np = waveform.numpy()[0]
            features = featurizer(waveform_np, sr)
            if isinstance(features, np.ndarray) and features.shape[0] != 768:
                logger.warning(f"特征维度 ({features.shape[0]}) 与模型期望 (768)不匹配，进行调整")
                if features.shape[0] < 768:
                    padded_features = np.zeros(768, dtype=features.dtype)
                    padded_features[:features.shape[0]] = features
                    features = padded_features
                else:
                    features = features[:768]
            return torch.from_numpy(features).float()
        except Exception as e:
            logger.error(f"提取特征时出错 ({audio_path}): {str(e)}")
            return None

    def load_model(model_path, device):
        logger.info("正在加载模型...")
        model = BaseModel().to(device)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            logger.info("使用 weights_only=True 加载模型成功")
        except TypeError:
            logger.warning("当前 PyTorch 版本不支持 weights_only 参数，使用默认参数加载模型")
            checkpoint = torch.load(model_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        model.eval()
        return model

    def predict_probs(audio_path, predictor):
        """使用MSERPredictor预测音频的情绪概率分布
        
        参数:
        audio_path: 音频文件路径
        predictor: MSERPredictor实例
        
        返回:
        numpy.ndarray: 各情绪的概率分布，或"SILENT"表示静音，或None表示错误
        """
        try:
            # 使用extract_features进行静音检测和特征提取，避免重复处理
            features = extract_features(audio_path, predictor._audio_featurizer)
            
            # 如果是静音文件，直接返回
            if features == "SILENT":
                return "SILENT"
            
            # 如果特征提取失败，返回None
            if features is None:
                return None
            
            # 使用MSERPredictor进行预测，获取完整的概率分布
            if hasattr(predictor, 'use_ms_model') and predictor.use_ms_model is not None:
                # 使用ModelScope模型
                labels, scores = predictor.predict(audio_path)
                # 需要将结果转换为概率分布格式
                probs = np.zeros(len(EMOTION_LABELS))
                for i, label in enumerate(EMOTION_LABELS):
                    if label in labels:
                        idx = labels.index(label)
                        probs[i] = scores[idx]
                return probs
            else:
                # 使用本地模型进行预测
                features = features.unsqueeze(0).to(predictor.device)
                # 执行预测
                output = predictor.predictor(features)
                result = torch.nn.functional.softmax(output, dim=-1)[0]
                result = result.data.cpu().numpy()
                return result
                
        except Exception as e:
            logger.error(f"预测音频 {audio_path} 时出错: {str(e)}")
            return None

    def fuse_emotion_results(emotion_probs_list):
        """置信度加权平均的情绪融合策略"""
        if not emotion_probs_list:
            return None
        
        if len(emotion_probs_list) == 1:
            # 只有一个结果，直接返回
            probs = emotion_probs_list[0]
            emotion_idx = np.argmax(probs)
            emotion_label = EMOTION_LABELS[emotion_idx]
            emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, probs)}
            return {"情绪标签": emotion_label, "各项情绪得分": emotion_scores}
        
        # 将概率列表转换为结果列表，包含置信度信息
        results = []
        for probs in emotion_probs_list:
            emotion_idx = np.argmax(probs)
            confidence = float(np.max(probs))
            results.append({
                'emotion': EMOTION_LABELS[emotion_idx],
                'confidence': confidence,
                'probabilities': probs
            })
        
        # 使用置信度加权平均融合
        return fuse_emotion_results_confidence_weighted(results)

    def apply_smoothing(probs, smoothing_factor=0.05):
        """对概率分布应用平滑处理，避免极端的100%单一情绪
        
        参数:
        probs: 原始概率分布
        smoothing_factor: 平滑因子，控制平滑程度
        
        返回:
        平滑后的概率分布
        """
        # 确保输入是numpy数组
        probs = np.array(probs)
        
        # 计算均匀分布的权重
        uniform_dist = np.ones(len(probs)) / len(probs)
        
        # 线性插值：(1-α) * 原分布 + α * 均匀分布
        smoothed_probs = (1 - smoothing_factor) * probs + smoothing_factor * uniform_dist
        
        # 确保概率和为1
        smoothed_probs = smoothed_probs / np.sum(smoothed_probs)
        
        return smoothed_probs

    def fuse_emotion_results_confidence_weighted(results):
        """置信度加权平均的融合方法
        
        这种方法根据置信度对不同预测结果进行加权平均，
        保持原始的情绪分布，不做任何平滑处理
        """
        if not results:
            return None
        
        # 提取所有概率分布和置信度
        all_probs = np.array([r['probabilities'] for r in results])
        confidences = np.array([r['confidence'] for r in results])
        
        # 计算置信度权重
        confidence_weights = confidences / np.sum(confidences)
        
        # 加权平均，不做平滑处理
        final_probs = np.sum(all_probs * confidence_weights[:, np.newaxis], axis=0)
        
        # 确定主要情绪
        emotion_idx = np.argmax(final_probs)
        emotion_label = EMOTION_LABELS[emotion_idx]

def predict_emotion_from_audio(audio_path, model_path=None, device='cpu'):
    """
    对单个音频文件进行情绪分析
    
    参数:
    audio_path: 音频文件路径
    model_path: 模型路径（可选）
    device: 设备类型（'cpu', 'cuda'）
    
    返回:
    包含各情绪得分的字典，失败时返回None
    """
    try:
        # 设备选择逻辑
        use_gpu = device == 'cuda' and torch.cuda.is_available()
        
        # 创建MSERPredictor实例
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'configs', 'base_model.yml')
        
        # 如果没有指定model_path，使用默认路径
        if model_path is None:
            default_model_path = os.path.join(current_dir, 'models', 'iic', 'BaseModel_Emotion2Vec', 'best_model')
            if os.path.exists(default_model_path):
                model_path = default_model_path
            else:
                logger.error(f"默认模型路径不存在: {default_model_path}")
                return None
        
        # 验证模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"指定的模型路径不存在: {model_path}")
            return None
        
        # 创建预测器
        predictor = MSERPredictor(
            configs=config_path,
            model_path=model_path,
            use_gpu=use_gpu,
            log_level="error"  # 减少日志输出
        )
        
        # 进行预测
        probs = predict_probs(audio_path, predictor)
        
        if isinstance(probs, str) and probs == "SILENT":
            logger.warning(f"音频文件 {audio_path} 被检测为静音")
            return None
        elif probs is not None:
            # 转换为情绪得分字典
            emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, probs)}
            return emotion_scores
        else:
            logger.error(f"无法处理音频文件: {audio_path}")
            return None
            
    except Exception as e:
        logger.error(f"分析音频文件 {audio_path} 时出错: {str(e)}")
        return None


def predict_emotions_batch(audio_paths, model_path=None, device='cpu'):
    """
    批量处理多个音频文件进行情绪分析
    
    参数:
    audio_paths: 音频文件路径列表
    model_path: 模型路径（可选）
    device: 设备类型（'cpu', 'cuda'）
    
    返回:
    包含各音频文件情绪得分字典的列表，失败或静音时对应位置为None
    """
    if not audio_paths:
        return []
    
    try:
        # 设备选择逻辑
        use_gpu = device == 'cuda' and torch.cuda.is_available()
        
        # 创建MSERPredictor实例
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'configs', 'base_model.yml')
        
        # 如果没有指定model_path，使用默认路径
        if model_path is None:
            default_model_path = os.path.join(current_dir, 'models', 'iic', 'BaseModel_Emotion2Vec', 'best_model')
            if os.path.exists(default_model_path):
                model_path = default_model_path
            else:
                logger.error(f"默认模型路径不存在: {default_model_path}")
                return [None] * len(audio_paths)
        
        # 验证模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"指定的模型路径不存在: {model_path}")
            return [None] * len(audio_paths)
        
        # 创建预测器（只创建一次）
        predictor = MSERPredictor(
            configs=config_path,
            model_path=model_path,
            use_gpu=use_gpu,
            log_level="error"  # 减少日志输出
        )
        
        # 第一步：静音检测，筛选出有效的音频文件
        results = []
        valid_audio_paths = []
        valid_indices = []
        silent_files = []
        
        print(f"开始处理 {len(audio_paths)} 个音频文件...")
        
        for i, audio_path in enumerate(audio_paths):
            try:
                # 检查文件是否存在
                if not os.path.exists(audio_path):
                    logger.warning(f"音频文件不存在: {audio_path}")
                    results.append(None)
                    continue
                
                # 静音检测（不提取特征，只检测是否静音）
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    waveform = process_audio(waveform, sr)
                    
                    if is_silent(waveform[0]):
                        logger.info(f"音频文件 {audio_path} 被检测为静音")
                        silent_files.append(audio_path)
                        results.append(None)
                        continue
                except Exception as e:
                    logger.error(f"静音检测失败 {audio_path}: {str(e)}")
                    results.append(None)
                    continue
                
                # 如果不是静音，添加到有效音频列表
                valid_audio_paths.append(audio_path)
                valid_indices.append(i)
                results.append(None)  # 占位符，稍后填充
                    
            except Exception as e:
                logger.error(f"处理音频文件 {audio_path} 时出错: {str(e)}")
                results.append(None)
        
        # 第二步：批量预测（如果有有效的音频文件）
        if valid_audio_paths:
            print(f"对 {len(valid_audio_paths)} 个有效音频进行批量预测...")
            try:
                # 使用MSERPredictor的predict_batch方法，传入音频路径
                batch_labels, batch_scores = predictor.predict_batch(valid_audio_paths)
                
                # 将批量预测结果填充到对应位置
                for j, (valid_idx, label, score) in enumerate(zip(valid_indices, batch_labels, batch_scores)):
                    try:
                        # 获取完整的概率分布
                        probs = predict_probs(valid_audio_paths[j], predictor)
                        
                        if isinstance(probs, str) and probs == "SILENT":
                            results[valid_idx] = None
                        elif probs is not None and len(probs) == len(EMOTION_LABELS):
                            # 转换为情绪得分字典
                            emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, probs)}
                            results[valid_idx] = emotion_scores
                        else:
                            logger.error(f"预测结果异常，索引: {valid_idx}")
                            results[valid_idx] = None
                    except Exception as e:
                        logger.error(f"处理预测结果失败，索引 {valid_idx}: {str(e)}")
                        results[valid_idx] = None
                        
            except Exception as e:
                logger.error(f"批量预测时出错: {str(e)}")
                # 如果批量预测失败，回退到单个预测
                print("批量预测失败，回退到单个预测模式...")
                for j, (valid_idx, audio_path) in enumerate(zip(valid_indices, valid_audio_paths)):
                    try:
                        # 单个预测
                        probs = predict_probs(audio_path, predictor)
                        
                        if isinstance(probs, str) and probs == "SILENT":
                            results[valid_idx] = None
                        elif probs is not None:
                            emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, probs)}
                            results[valid_idx] = emotion_scores
                        else:
                            results[valid_idx] = None
                            
                    except Exception as e:
                        logger.error(f"单个预测失败，索引 {valid_idx}: {str(e)}")
                        results[valid_idx] = None
        
        # 统计结果
        valid_count = sum(1 for r in results if r is not None)
        silent_count = len(silent_files)
        print(f"批量处理完成: {valid_count} 个有效结果, {silent_count} 个静音文件")
        
        return results
            
    except Exception as e:
        logger.error(f"批量分析音频文件时出错: {str(e)}")
        return [None] * len(audio_paths)


def fuse_emotion_results_confidence_weighted_smooth(results):
        """置信度加权加平滑处理的融合方法（备用）
        
        这种方法结合了以下策略：
        1. 根据置信度对不同预测结果进行加权
        2. 应用平滑处理避免极端分布
        3. 确保每种情绪都有最小概率
        """
        if not results:
            return None
        
        # 提取所有概率分布和置信度
        all_probs = np.array([r['probabilities'] for r in results])
        confidences = np.array([r['confidence'] for r in results])
        
        # 计算置信度权重（使用平方根来减少极端权重差异）
        confidence_weights = np.sqrt(confidences)
        confidence_weights = confidence_weights / np.sum(confidence_weights)
        
        # 加权平均
        weighted_avg_probs = np.sum(all_probs * confidence_weights[:, np.newaxis], axis=0)
        
        # 应用平滑处理
        smoothed_probs = apply_smoothing(weighted_avg_probs, smoothing_factor=0.08)
        
        # 确保每种情绪至少有最小概率（1%）
        min_prob = 0.01
        smoothed_probs = np.maximum(smoothed_probs, min_prob)
        
        # 重新归一化
        smoothed_probs = smoothed_probs / np.sum(smoothed_probs)
        
        # 确定主要情绪
        emotion_idx = np.argmax(smoothed_probs)
        emotion_label = EMOTION_LABELS[emotion_idx]
        
        # 转换为百分比
        emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, smoothed_probs)}
        
        # 计算平均置信度
        avg_confidence = np.mean(confidences)
        
        print(f"使用置信度加权平滑融合: {emotion_label} (平均置信度: {avg_confidence:.3f})")
        print(f"融合了 {len(results)} 个预测结果")
        
        return {
            "情绪标签": emotion_label,
            "各项情绪得分": emotion_scores
        }

def fuse_emotion_results_voting(results):
    """基于投票的融合方法"""
    # 统计每种情绪的投票数（加权投票）
    emotion_votes = {}
    total_weight = 0
    
    for result in results:
        emotion = result['emotion']
        weight = result['confidence']
        
        if emotion not in emotion_votes:
            emotion_votes[emotion] = 0
        emotion_votes[emotion] += weight
        total_weight += weight
    
    # 找到得票最多的情绪
    winning_emotion = max(emotion_votes.keys(), key=lambda x: emotion_votes[x])
    
    # 计算该情绪的平均概率
    emotion_probs = []
    for result in results:
        if result['emotion'] == winning_emotion:
            emotion_probs.append(result['probabilities'])
    
    if emotion_probs:
        avg_probs = np.mean(emotion_probs, axis=0)
        confidence = emotion_votes[winning_emotion] / total_weight
    else:
        avg_probs = results[0]['probabilities']
        confidence = 0.5
    
    emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, avg_probs)}
    
    print(f"使用投票融合: {winning_emotion} (置信度: {confidence:.3f})")
    return {"情绪标签": winning_emotion, "各项情绪得分": emotion_scores}

def fuse_emotion_results_max_confidence(results):
        """选择置信度最高的单个预测结果"""
        # 找到置信度最高的结果
        max_confidence_result = max(results, key=lambda x: x['confidence'])
        
        emotion_scores = {label: float(round(prob * 100, 2)) for label, prob in zip(EMOTION_LABELS, max_confidence_result['probabilities'])}
        
        print(f"使用最高置信度融合: {max_confidence_result['emotion']} (置信度: {max_confidence_result['confidence']:.3f})")
        return {
            "情绪标签": max_confidence_result['emotion'],
            "各项情绪得分": emotion_scores
        }

def fuse_emotion_results_hierarchical(results):
    """分层融合：先按置信度分组，再融合（备用方法）"""
    if not results:
        return None
    
    # 按置信度分为高、中、低三组
    high_conf = [r for r in results if r['confidence'] > 0.8]
    mid_conf = [r for r in results if 0.5 <= r['confidence'] <= 0.8]
    low_conf = [r for r in results if r['confidence'] < 0.5]
    
    # 优先使用高置信度结果
    if high_conf:
        target_results = high_conf
        method = '高置信度融合'
    elif mid_conf:
        target_results = mid_conf
        method = '中置信度融合'
    else:
        target_results = low_conf
        method = '低置信度融合'
    
    print(f"使用{method}，处理{len(target_results)}个结果")
    
    # 对选定组进行加权平均
    return fuse_emotion_results_weighted_avg(target_results)

def fuse_emotion_results_weighted_avg(results):
    """传统的加权平均方法（备用）"""
    if not results:
        return None
    
    all_probs = np.array([r['probabilities'] for r in results])
    confidence_weights = np.array([r['confidence'] for r in results])
    
    sum_confidence_weights = np.sum(confidence_weights)
    normalized_weights = (confidence_weights / sum_confidence_weights) if sum_confidence_weights > 0 else (np.ones_like(confidence_weights) / len(confidence_weights))
    
    weighted_avg_probs = np.sum(all_probs * normalized_weights[:, np.newaxis], axis=0)
    emotion_idx = np.argmax(weighted_avg_probs)
    emotion_label = EMOTION_LABELS[emotion_idx]
    emotion_scores = {label: round(prob * 100, 2) for label, prob in zip(EMOTION_LABELS, weighted_avg_probs)}
    
    print(f"使用加权平均融合: {emotion_label}")
    return {"情绪标签": emotion_label, "各项情绪得分": emotion_scores}
    


def calculate_scale_results_by_labels(fused_results_by_label):
    """
    根据不同标签的情绪得分分别计算三个维度的量表结果。
    每个维度直接使用对应标签的情绪得分，不进行融合。
    为不同维度设置不同的权重系统以增加区分度。
    
    参数:
    fused_results_by_label: dict, 格式为 {标签: 情绪得分字典}
    
    返回:
    包含三个维度得分的字典
    """
    
    # 统一的情绪权重系统（所有维度使用相同权重）
    emotion_weights = {
        '快乐': 1,
        '惊讶': 1,
        '平静': 0.5,
        '中性': 0,
        '恐惧': -1,
        '悲伤': -1,
        '愤怒': -1,
        '厌恶': -1
    }
    
    def calculate_weighted_score(emotion_scores, weights):
        """
        计算加权总分并归一化到0-100范围
        """
        if not emotion_scores:
            return 50.0  # 默认中性值
            
        # 计算加权总分
        total_score = 0
        for emotion, score in emotion_scores.items():
            weight = weights.get(emotion, 0)
            total_score += score * weight
        
        # 计算理论最小值和最大值
        t_min = sum([100 * weight for weight in weights.values() if weight < 0])
        t_max = sum([100 * weight for weight in weights.values() if weight > 0])
        
        # 归一化到0-100范围
        if t_max - t_min != 0:
            normalized_score = ((total_score - t_min) / (t_max - t_min)) * 100
        else:
            normalized_score = 50
        
        # 确保结果在0-100范围内
        normalized_score = max(0, min(100, normalized_score))
        
        return round(normalized_score, 2)
    
    results = {}
    
    # 直接使用对应标签的情绪得分计算各维度（使用统一权重系统）
    # 个性画像：使用 '个性画像' 标签
    if '个性画像' in fused_results_by_label:
        emotion_scores = fused_results_by_label['个性画像']
        personality_score = calculate_weighted_score(emotion_scores, emotion_weights)
        results['个性画像'] = personality_score
        print(f"维度 '个性画像' 情绪得分: {emotion_scores}")
        print(f"维度 '个性画像' 计算完成，得分: {personality_score}")
    else:
        results['个性画像'] = None
        print(f"警告: 未找到 '个性画像' 标签的有效数据，该维度值为空")
    
    # 支持系统：使用 '支持系统' 标签
    if '支持系统' in fused_results_by_label:
        emotion_scores = fused_results_by_label['支持系统']
        support_score = calculate_weighted_score(emotion_scores, emotion_weights)
        results['支持系统'] = support_score
        print(f"维度 '支持系统' 情绪得分: {emotion_scores}")
        print(f"维度 '支持系统' 计算完成，得分: {support_score}")
    else:
        results['支持系统'] = None
        print(f"警告: 未找到 '支持系统' 标签的有效数据，该维度值为空")
    
    # 职场感知：使用 '职场感知' 标签
    if '职场感知' in fused_results_by_label:
        emotion_scores = fused_results_by_label['职场感知']
        work_score = calculate_weighted_score(emotion_scores, emotion_weights)
        results['职场感知'] = work_score
        print(f"维度 '职场感知' 情绪得分: {emotion_scores}")
        print(f"维度 '职场感知' 计算完成，得分: {work_score}")
    else:
        results['职场感知'] = None
        print(f"警告: 未找到 '职场感知' 标签的有效数据，该维度值为空")
    
    return results

    def load_user_report(json_path):
        """加载用户综合报告JSON文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载用户报告文件失败: {str(e)}")
            return None

    def get_audio_files_by_user(user_report, user_id, data_base_path, exclude_labels=None, include_labels=None):
        """根据用户ID从报告中获取音频文件路径，支持标签过滤"""
        if user_id not in user_report:
            print(f"错误: 用户ID '{user_id}' 在报告中不存在")
            return []
        
        audio_data = user_report[user_id].get('音频数据', [])
        filtered_audio_files = []
        
        for audio_info in audio_data:
            label = audio_info.get('标签', '')
            
            # 排除特定标签
            if exclude_labels and label in exclude_labels:
                continue
            
            # 只包含特定标签
            if include_labels and label not in include_labels:
                continue
            
            file_path = audio_info.get('文件地址', '').strip()
            if file_path:
                # 构建完整的文件路径
                full_path = os.path.join(data_base_path, file_path.lstrip('/'))
                if os.path.exists(full_path):
                    filtered_audio_files.append({
                        'path': full_path,
                        'label': label,
                        'title': audio_info.get('题目', ''),
                        'filename': os.path.basename(full_path)
                    })
                else:
                    print(f"警告: 音频文件不存在: {full_path}")
        
        return filtered_audio_files

    def save_results(id_str, results_dict, output_dir):
        """只保存JSON文件，不保存CSV文件"""
        if not DEPENDENCIES_AVAILABLE:
            print("错误：缺少依赖库，无法保存结果")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{id_str}_结果.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        return json_path

    def predict_user_emotion_in_memory(user_id, **kwargs):
        """
        在内存中执行用户情绪分析，返回结果字典而不保存JSON文件
        
        Args:
            user_id (str): 用户ID
            **kwargs: 其他可选参数
                - data_dir: 数据目录路径
                - user_report: 用户综合报告JSON文件路径
                - model_path: 模型路径
                - output_dir: 输出目录
                - device: 设备类型 (cpu/cuda)
                - exclude_labels: 排除的标签列表
                - include_labels: 包含的标签列表
        
        Returns:
            dict: 情绪分析结果字典，失败时返回None
        """
        try:
            # 设置默认参数
            if config_available:
                default_data_dir = str(DATA1_PATH)
                default_user_report = str(DATA1_PATH.parent / 'data' / '用户综合报告.json')
                default_model_path = str(MODEL_PATH / 'iic' / 'BaseModel_Emotion2Vec' / 'best_model' / 'model.pth')
                default_output_dir = str(RESULTS_PATH)
            else:
                default_data_dir = os.path.join(current_dir, 'data1')
                default_user_report = os.path.join(current_dir, '..', 'data', '用户综合报告.json')
                default_model_path = os.path.join(current_dir, 'models', 'iic', 'BaseModel_Emotion2Vec', 'best_model', 'model.pth')
                default_output_dir = os.path.join(current_dir, 'results')
            
            data_dir = kwargs.get('data_dir', default_data_dir)
            user_report_path = kwargs.get('user_report', default_user_report)
            model_path = kwargs.get('model_path', default_model_path)
            output_dir = kwargs.get('output_dir', default_output_dir)
            device = kwargs.get('device', 'cpu')
            exclude_labels = kwargs.get('exclude_labels', ['预热题'])
            include_labels = kwargs.get('include_labels', None)
            
            print(f"开始为用户 '{user_id}' 进行情绪分析...")
            
            # 设备配置
            use_gpu = device == 'cuda' and torch.cuda.is_available()
            device_obj = torch.device('cuda' if use_gpu else 'cpu')
            print(f"使用设备: {device_obj}")
            
            # 加载用户报告
            user_report = load_user_report(user_report_path)
            if not user_report:
                print(f"错误: 无法加载用户报告文件 {user_report_path}")
                return None
            
            # 检查用户ID是否存在
            if user_id not in user_report:
                print(f"错误: 指定的用户ID '{user_id}' 在报告中不存在")
                return None
            
            # 创建MSERPredictor实例
            try:
                config_path = os.path.join(current_dir, 'configs', 'base_model.yml')
                predictor = MSERPredictor(
                    configs=config_path,
                    model_path=model_path,
                    use_gpu=use_gpu,
                    log_level="info"
                )
                print(f"成功创建MSERPredictor，类别标签: {predictor.class_labels}")
            except Exception as e:
                print(f"创建MSERPredictor失败: {str(e)}")
                return None
            
            # 获取音频文件
            audio_files = get_audio_files_by_user(
                user_report, 
                user_id, 
                data_dir,
                exclude_labels=exclude_labels,
                include_labels=include_labels
            )
            
            if not audio_files:
                print(f"用户 '{user_id}' 没有符合条件的音频文件")
                return None
            
            print(f"找到 {len(audio_files)} 个符合条件的音频文件")
            
            # 按标签分组存储情绪预测结果
            emotion_results_by_label = {}
            emotion_probs_list = []
            silent_files = []
            
            # 处理音频文件
            for audio_info in audio_files:
                audio_file = audio_info['path']
                label = audio_info['label']
                probs = predict_probs(audio_file, predictor)
                
                if isinstance(probs, str) and probs == "SILENT":
                    silent_files.append(audio_info)
                    continue
                elif probs is not None:
                    emotion_probs_list.append(probs)
                    
                    if label not in emotion_results_by_label:
                        emotion_results_by_label[label] = []
                    emotion_results_by_label[label].append(probs)
            
            if not emotion_probs_list:
                print(f"警告: 用户 '{user_id}' 的所有音频文件都是静音或无效，无法进行情绪分析。")
                return {
                    "融合情绪结果": {
                        "情绪标签": None,
                        "各项情绪得分": {label: None for label in EMOTION_LABELS}
                    },
                    "三维度结果": {
                        "个性画像": None,
                        "支持系统": None,
                        "职场感知": None
                    },
                    "按题目分组结果": {}
                }
            
            # 计算总体融合结果
            fused_result = fuse_emotion_results(emotion_probs_list)
            if not fused_result:
                print(f"错误: 未能融合用户 '{user_id}' 的情绪结果。")
                return None
            
            # 计算各标签的融合结果
            fused_results_by_label = {}
            for label, probs_list in emotion_results_by_label.items():
                label_fused = fuse_emotion_results(probs_list)
                if label_fused:
                    fused_results_by_label[label] = label_fused['各项情绪得分']
            
            # 计算三维度结果
            scale_results = calculate_scale_results_by_labels(fused_results_by_label)
            
            overall_results = {
                "融合情绪结果": fused_result, 
                "三维度结果": scale_results,
                "按题目分组结果": fused_results_by_label
            }
            
            print(f"用户 '{user_id}' 的情绪分析完成")
            return overall_results
            
        except Exception as e:
            print(f"内存中情绪分析失败: {e}")
            return None

def main():
    if not DEPENDENCIES_AVAILABLE:
        print("错误：缺少必要的依赖库，请安装以下库:")
        for dep in MISSING_DEPENDENCIES: print(f"  - {dep}")
        return
    
    parser = argparse.ArgumentParser(description='使用Emotion2Vec特征预测音频情绪')
    # 使用配置文件中的路径作为默认值
    if config_available:
        default_data_dir = str(DATA1_PATH)
        default_user_report = str(DATA1_PATH.parent / 'data' / '用户综合报告.json')
        default_model_path = str(MODEL_PATH / 'iic' / 'BaseModel_Emotion2Vec' / 'best_model' / 'model.pth')
        default_output_dir = str(RESULTS_PATH)
    else:
        # 如果config不可用，使用相对路径
        default_data_dir = os.path.join(current_dir, 'data1')
        default_user_report = os.path.join(current_dir, '..', 'data', '用户综合报告.json')
        default_model_path = os.path.join(current_dir, 'models', 'iic', 'BaseModel_Emotion2Vec', 'best_model', 'model.pth')
        default_output_dir = os.path.join(current_dir, 'results')
    
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='数据目录路径')
    parser.add_argument('--user_report', type=str, default=default_user_report, help='用户综合报告JSON文件路径')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='模型路径')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='输出目录')
    parser.add_argument('--id', type=str, help='指定用户ID')
    parser.add_argument('--exclude_labels', type=str, nargs='*', default=['预热题'], help='排除的标签列表')
    parser.add_argument('--include_labels', type=str, nargs='*', help='只包含的标签列表（如果指定，则只处理这些标签的音频）')
    parser.add_argument('--device', type=str, default='cpu', help='设备: cpu或cuda')
    args = parser.parse_args()

    # 设备配置
    use_gpu = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载用户报告
    user_report = load_user_report(args.user_report)
    if not user_report:
        print(f"错误: 无法加载用户报告文件 {args.user_report}")
        return
    
    # 获取可用的用户ID列表
    available_users = list(user_report.keys())
    if not available_users:
        print("错误: 用户报告中没有找到任何用户数据")
        return
    
    # 选择用户ID
    if args.id:
        if args.id not in available_users:
            print(f"错误: 指定的用户ID '{args.id}' 在报告中不存在")
            print(f"可用的用户ID: {', '.join(available_users)}")
            return
        selected_user_id = args.id
        print(f"处理指定的用户ID: {selected_user_id}")
    else:
        selected_user_id = random.choice(available_users)
        print(f"未指定用户ID，随机选择: {selected_user_id}")
    
    # 创建MSERPredictor实例
    try:
        # 获取配置文件的完整路径
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'configs', 'base_model.yml')
        
        # 使用完整路径创建预测器
        predictor = MSERPredictor(
            configs=config_path,  # 使用完整的配置文件路径
            model_path=args.model_path,
            use_gpu=use_gpu,
            log_level="info"
        )
        print(f"成功创建MSERPredictor，类别标签: {predictor.class_labels}")
        print(f"特征维度: {predictor._audio_featurizer.feature_dim}")
    except Exception as e:
        print(f"创建MSERPredictor失败: {str(e)}")
        return
    
    # 获取音频文件
    audio_files = get_audio_files_by_user(
        user_report, 
        selected_user_id, 
        args.data_dir,
        exclude_labels=args.exclude_labels,
        include_labels=args.include_labels
    )
    
    if not audio_files:
        print(f"用户 '{selected_user_id}' 没有符合条件的音频文件")
        if args.exclude_labels:
            print(f"排除的标签: {', '.join(args.exclude_labels)}")
        if args.include_labels:
            print(f"包含的标签: {', '.join(args.include_labels)}")
        return
    
    print(f"找到 {len(audio_files)} 个符合条件的音频文件 (用户ID: {selected_user_id}):")
    for audio in audio_files:
        print(f"  - {audio['filename']} (标签: {audio['label']})")
    
    if args.exclude_labels:
        print(f"已排除标签: {', '.join(args.exclude_labels)}")
    if args.include_labels:
        print(f"只包含标签: {', '.join(args.include_labels)}")



    # 按标签分组存储情绪预测结果
    emotion_results_by_label = {}
    emotion_probs_list = []  # 保留用于总体融合
    silent_files = []  # 记录静音文件
    
    print(f"\n开始处理用户 '{selected_user_id}' 的音频...")
    for audio_info in audio_files:
        audio_file = audio_info['path']
        label = audio_info['label']
        probs = predict_probs(audio_file, predictor)
        
        if isinstance(probs, str) and probs == "SILENT":
            silent_files.append(audio_info)
            print(f"跳过静音文件: {audio_info['filename']} (标签: {audio_info['label']})")
            continue
        elif probs is not None:
            emotion_probs_list.append(probs)
            
            # 按标签分组存储
            if label not in emotion_results_by_label:
                emotion_results_by_label[label] = []
            emotion_results_by_label[label].append(probs)
            
            print(f"成功处理文件: {audio_info['filename']} (标签: {audio_info['label']})")
        else:
            print(f"处理文件 {audio_info['filename']} 时未能提取特征或预测。")
    
    # 输出静音检测统计
    print(f"\n=== 静音检测统计 ===")
    print(f"总音频文件数: {len(audio_files)}")
    print(f"静音文件数: {len(silent_files)}")
    print(f"有效音频文件数: {len(emotion_probs_list)}")
    
    if silent_files:
        print(f"\n跳过的静音文件列表:")
        for silent_file in silent_files:
            print(f"  - {silent_file['filename']} (标签: {silent_file['label']})")
    else:
        print("\n未发现静音文件")

    if not emotion_probs_list:
        if silent_files and len(silent_files) == len(audio_files):
            print(f"\n警告: 用户 '{selected_user_id}' 的所有音频文件都是静音，无法进行情绪分析。")
            # 创建空的融合结果
            fused_result = {
                "情绪标签": None,
                "各项情绪得分": {label: None for label in EMOTION_LABELS}
            }
            fused_results_by_label = {}  # 空的标签结果
        else:
            print(f"错误: 未能从用户 '{selected_user_id}' 的音频中提取任何有效的情绪概率。")
            return
    else:
        print(f"\n开始融合 {len(emotion_probs_list)} 个有效音频的情绪结果...")
        # 计算总体融合结果
        fused_result = fuse_emotion_results(emotion_probs_list)
        if not fused_result:
            print(f"错误: 未能融合用户 '{selected_user_id}' 的情绪结果。")
            return
    
    # 计算各标签的融合结果（只在有非静音音频时执行）
    if emotion_probs_list:  # 只有当有非静音音频时才计算
        print(f"\n=== 按标签融合结果 ===")
        fused_results_by_label = {}
        for label, probs_list in emotion_results_by_label.items():
            label_fused = fuse_emotion_results(probs_list)
            if label_fused:
                fused_results_by_label[label] = label_fused['各项情绪得分']
                print(f"标签 '{label}': 融合完成，使用 {len(probs_list)} 个有效音频")
            else:
                print(f"标签 '{label}': 融合失败")
        
        # 检查哪些标签因为全部静音而无数据
        all_labels = set(audio_info['label'] for audio_info in audio_files)
        silent_labels = set(silent_file['label'] for silent_file in silent_files)
        for label in all_labels:
            if label not in emotion_results_by_label:
                label_silent_count = sum(1 for sf in silent_files if sf['label'] == label)
                label_total_count = sum(1 for af in audio_files if af['label'] == label)
                if label_silent_count == label_total_count:
                    print(f"标签 '{label}': 所有 {label_total_count} 个音频均为静音，无法分析")
                else:
                    print(f"标签 '{label}': 处理异常，{label_total_count} 个音频中 {label_silent_count} 个静音")
        
    scale_results = calculate_scale_results_by_labels(fused_results_by_label)
    
    overall_results = {
        "融合情绪结果": fused_result, 
        "三维度结果": scale_results,
        "按题目分组结果": fused_results_by_label
    }
    
    # 打印中间结果到终端
    print("\n=== 中间结果 ===")
    print(json.dumps(overall_results, ensure_ascii=False, indent=2))
    
    json_path = save_results(selected_user_id, overall_results, args.output_dir)
    if json_path:
        print(f"\n结果已保存到: {json_path}")
    
    # 返回结果用于直接调用
    return overall_results

    print("\n=== 用户 '{}' 的融合情绪预测结果 ===".format(selected_user_id))
    
    if fused_result.get('情绪标签') is None:
        print("  情绪标签: 无数据(所有音频均为静音)")
        print("  各项情绪得分: 无数据(所有音频均为静音)")
    else:
        print(f"  情绪标签: {fused_result.get('情绪标签', '未知')} (基于 {len(emotion_probs_list)} 个有效音频)")
        sorted_scores = sorted(fused_result.get('各项情绪得分', {}).items(), key=lambda item: item[1] if item[1] is not None else 0, reverse=True)
        for emotion_name, score in sorted_scores: 
            if score is not None:
                print(f"    - {emotion_name}: {score:.4f}")
            else:
                print(f"    - {emotion_name}: 无数据")
    
    print("\n=== 用户 '{}' 的多维评估 ===".format(selected_user_id))
    
    # 格式化输出，将None显示为"无数据(静音)"
    def format_score(score):
        if score is None:
            return "无数据(静音)"
        return f"{score:.2f}"
    
    print(f"  个性画像: {format_score(scale_results.get('个性画像', 'N/A'))}")
    print(f"  支持系统: {format_score(scale_results.get('支持系统', 'N/A'))}")
    print(f"  职场感知: {format_score(scale_results.get('职场感知', 'N/A'))}")

def analyze_emotion_for_user(data_dir, user_report_path, user_id, model_path=None, device='auto', exclude_labels=None, include_labels=None):
    """
    直接进行情绪分析并返回结果，不保存文件
    
    参数:
    data_dir: 音频数据目录
    user_report_path: 用户报告JSON文件路径
    user_id: 用户ID
    model_path: 模型路径（可选）
    device: 设备类型（'auto', 'cpu', 'cuda'）
    exclude_labels: 排除的标签列表
    include_labels: 包含的标签列表
    
    返回:
    包含情绪分析结果的字典，格式与保存的JSON文件相同
    """
    import random
    
    # 设备选择逻辑
    if device == 'auto':
        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
    else:
        use_gpu = device == 'cuda'
    
    print(f"使用设备: {device}")
    
    # 加载用户报告
    user_report = load_user_report(user_report_path)
    if not user_report:
        print(f"错误: 无法加载用户报告文件 {user_report_path}")
        return None
    
    # 获取可用的用户ID列表
    available_users = list(user_report.keys())
    if not available_users:
        print("错误: 用户报告中没有找到任何用户数据")
        return None
    
    # 检查用户ID
    if user_id not in available_users:
        print(f"错误: 指定的用户ID '{user_id}' 在报告中不存在")
        print(f"可用的用户ID: {', '.join(available_users)}")
        return None
    
    print(f"处理指定的用户ID: {user_id}")
    
    # 创建MSERPredictor实例
    try:
        # 获取配置文件的完整路径
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'configs', 'base_model.yml')
        
        # 如果没有指定model_path，使用默认路径
        if model_path is None:
            # 使用相对于当前脚本的默认模型路径
            default_model_path = os.path.join(current_dir, 'models', 'iic', 'BaseModel_Emotion2Vec', 'best_model')
            if os.path.exists(default_model_path):
                model_path = default_model_path
                print(f"使用默认模型路径: {model_path}")
            else:
                print(f"错误: 默认模型路径不存在: {default_model_path}")
                print("请指定正确的模型路径或确保默认模型文件存在")
                return None
        
        # 验证模型路径是否存在
        if not os.path.exists(model_path):
            print(f"错误: 指定的模型路径不存在: {model_path}")
            return None
        
        # 使用完整路径创建预测器
        predictor = MSERPredictor(
            configs=config_path,  # 使用完整的配置文件路径
            model_path=model_path,
            use_gpu=use_gpu,
            log_level="info"
        )
        print(f"成功创建MSERPredictor，类别标签: {predictor.class_labels}")
        print(f"特征维度: {predictor._audio_featurizer.feature_dim}")
    except Exception as e:
        print(f"创建MSERPredictor失败: {str(e)}")
        return None
    
    # 获取音频文件
    audio_files = get_audio_files_by_user(
        user_report, 
        user_id, 
        data_dir,
        exclude_labels=exclude_labels,
        include_labels=include_labels
    )
    
    if not audio_files:
        print(f"用户 '{user_id}' 没有符合条件的音频文件")
        if exclude_labels:
            print(f"排除的标签: {', '.join(exclude_labels)}")
        if include_labels:
            print(f"包含的标签: {', '.join(include_labels)}")
        return None
    
    print(f"找到 {len(audio_files)} 个符合条件的音频文件 (用户ID: {user_id}):")
    for audio in audio_files:
        print(f"  - {audio['filename']} (标签: {audio['label']})")
    
    if exclude_labels:
        print(f"已排除标签: {', '.join(exclude_labels)}")
    if include_labels:
        print(f"只包含标签: {', '.join(include_labels)}")
    
    # 按标签分组存储情绪预测结果
    emotion_results_by_label = {}
    emotion_probs_list = []  # 保留用于总体融合
    silent_files = []  # 记录静音文件
    
    print(f"\n开始处理用户 '{user_id}' 的音频...")
    for audio_info in audio_files:
        audio_file = audio_info['path']
        label = audio_info['label']
        probs = predict_probs(audio_file, predictor)
        
        if isinstance(probs, str) and probs == "SILENT":
            silent_files.append(audio_info)
            print(f"跳过静音文件: {audio_info['filename']} (标签: {audio_info['label']})")
            continue
        elif probs is not None:
            emotion_probs_list.append(probs)
            
            # 按标签分组存储
            if label not in emotion_results_by_label:
                emotion_results_by_label[label] = []
            emotion_results_by_label[label].append(probs)
            
            print(f"成功处理文件: {audio_info['filename']} (标签: {audio_info['label']})")
        else:
            print(f"处理文件 {audio_info['filename']} 时未能提取特征或预测。")
    
    # 输出静音检测统计
    print(f"\n=== 静音检测统计 ===")
    print(f"总音频文件数: {len(audio_files)}")
    print(f"静音文件数: {len(silent_files)}")
    print(f"有效音频文件数: {len(emotion_probs_list)}")
    
    if silent_files:
        print(f"\n跳过的静音文件列表:")
        for silent_file in silent_files:
            print(f"  - {silent_file['filename']} (标签: {silent_file['label']})")
    else:
        print("\n未发现静音文件")
    
    if not emotion_probs_list:
        if silent_files and len(silent_files) == len(audio_files):
            print(f"\n警告: 用户 '{user_id}' 的所有音频文件都是静音，无法进行情绪分析。")
            # 创建空的融合结果
            fused_result = {
                "情绪标签": None,
                "各项情绪得分": {label: None for label in EMOTION_LABELS}
            }
            fused_results_by_label = {}  # 空的标签结果
        else:
            print(f"错误: 未能从用户 '{user_id}' 的音频中提取任何有效的情绪概率。")
            return None
    else:
        print(f"\n开始融合 {len(emotion_probs_list)} 个有效音频的情绪结果...")
        # 计算总体融合结果
        fused_result = fuse_emotion_results(emotion_probs_list)
        if not fused_result:
            print(f"错误: 未能融合用户 '{user_id}' 的情绪结果。")
            return None
    
    # 计算各标签的融合结果（只在有非静音音频时执行）
    if emotion_probs_list:  # 只有当有非静音音频时才计算
        print(f"\n=== 按标签融合结果 ===")
        fused_results_by_label = {}
        for label, probs_list in emotion_results_by_label.items():
            label_fused = fuse_emotion_results(probs_list)
            if label_fused:
                fused_results_by_label[label] = label_fused['各项情绪得分']
                print(f"标签 '{label}': 融合完成，使用 {len(probs_list)} 个有效音频")
            else:
                print(f"标签 '{label}': 融合失败")
        
        # 检查哪些标签因为全部静音而无数据
        all_labels = set(audio_info['label'] for audio_info in audio_files)
        silent_labels = set(silent_file['label'] for silent_file in silent_files)
        for label in all_labels:
            if label not in emotion_results_by_label:
                label_silent_count = sum(1 for sf in silent_files if sf['label'] == label)
                label_total_count = sum(1 for af in audio_files if af['label'] == label)
                if label_silent_count == label_total_count:
                    print(f"标签 '{label}': 所有 {label_total_count} 个音频均为静音，无法分析")
                else:
                    print(f"标签 '{label}': 处理异常，{label_total_count} 个音频中 {label_silent_count} 个静音")
    
    scale_results = calculate_scale_results_by_labels(fused_results_by_label)
    
    overall_results = {
        "融合情绪结果": fused_result, 
        "三维度结果": scale_results,
        "按题目分组结果": fused_results_by_label
    }
    
    # 打印中间结果到终端
    print("\n=== 中间结果 ===")
    print(json.dumps(overall_results, ensure_ascii=False, indent=2))
    
    print("\n=== 用户 '{}' 的融合情绪预测结果 ===".format(user_id))
    
    if fused_result.get('情绪标签') is None:
        print("  情绪标签: 无数据(所有音频均为静音)")
        print("  各项情绪得分: 无数据(所有音频均为静音)")
    else:
        print(f"  情绪标签: {fused_result.get('情绪标签', '未知')} (基于 {len(emotion_probs_list)} 个有效音频)")
        sorted_scores = sorted(fused_result.get('各项情绪得分', {}).items(), key=lambda item: item[1] if item[1] is not None else 0, reverse=True)
        for emotion_name, score in sorted_scores: 
            if score is not None:
                print(f"    - {emotion_name}: {score:.4f}")
            else:
                print(f"    - {emotion_name}: 无数据")
    
    print("\n=== 用户 '{}' 的多维评估 ===".format(user_id))
    
    # 格式化输出，将None显示为"无数据(静音)"
    def format_score(score):
        if score is None:
            return "无数据(静音)"
        return f"{score:.2f}"
    
    print(f"  个性画像: {format_score(scale_results.get('个性画像', 'N/A'))}")
    print(f"  支持系统: {format_score(scale_results.get('支持系统', 'N/A'))}")
    print(f"  职场感知: {format_score(scale_results.get('职场感知', 'N/A'))}")
    
    return overall_results

if __name__ == '__main__':
    main()