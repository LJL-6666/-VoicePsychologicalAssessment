#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
朔黄工程数据处理工具
功能：从JSON格式的测评包数据中提取问卷结果和下载音频文件
作者：AI助手
日期：2025年
"""

import os
import json
import requests
import warnings
from datetime import datetime
import traceback
from urllib.parse import urlparse

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, temp_base_dir="./temp_data"):
        self.temp_base_dir = temp_base_dir
        self.logs = []
    
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.logs.append(log_message)
        print(log_message)
    
    def load_json_data(self, json_file_path):
        """
        加载JSON数据文件
        Args:
            json_file_path (str): JSON文件路径
        Returns:
            dict or None: 加载的数据，失败时返回None
        """
        self.log(f"开始加载JSON数据文件: {json_file_path}")
        
        if not os.path.exists(json_file_path):
            self.log(f"错误：JSON文件不存在 - {json_file_path}")
            return None
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.log("✓ JSON文件加载成功")
            return data
        except Exception as e:
            self.log(f"错误：加载JSON文件失败 - {e}")
            return None
    
    def extract_questionnaire_results(self, data):
        """
        从JSON数据中提取问卷结果
        Args:
            data (dict): JSON数据
        Returns:
            dict: 提取的问卷结果
        """
        self.log("开始提取问卷结果...")
        questionnaire_results = {}
        
        if 'data' not in data or 'dataList' not in data['data']:
            self.log("警告：未找到问卷数据列表")
            return questionnaire_results
        
        data_list = data['data']['dataList']
        self.log(f"找到 {len(data_list)} 项问卷数据")
        
        for item in data_list:
            questionnaire_name = item.get('questionnaireName', '')
            factor_name = item.get('facterName', '')
            score = item.get('score', '')
            score_str = item.get('scoreStr', '')
            level = item.get('level', '')
            feedback = item.get('feedback', '')
            
            # 构建问卷结果键名
            if questionnaire_name and factor_name:
                key = f"{questionnaire_name}_{factor_name}"
            elif questionnaire_name:
                key = questionnaire_name
            elif factor_name:
                key = factor_name
            else:
                continue
            
            questionnaire_results[key] = {
                'score': score,
                'scoreStr': score_str,
                'level': level,
                'feedback': feedback,
                'questionnaireName': questionnaire_name,
                'facterName': factor_name
            }
            
            self.log(f"  - {questionnaire_name}: {score} ({level})")
        
        # 添加个人信息
        if 'data' in data and 'personInfo' in data['data']:
            person_info = data['data']['personInfo']
            questionnaire_results['personInfo'] = person_info
            self.log(f"✓ 个人信息已添加")
        
        self.log(f"✓ 问卷结果提取完成，共 {len(questionnaire_results)} 项")
        return questionnaire_results
    
    def download_audio_file(self, url, save_path, timeout=30):
        """
        下载音频文件
        Args:
            url (str): 音频文件URL
            save_path (str): 保存路径
            timeout (int): 超时时间
        Returns:
            bool: 下载是否成功
        """
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 下载文件
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
        except Exception as e:
            self.log(f"下载失败: {e}")
            return False
    
    def process_audio_files(self, data, record_id, download_audio=True):
        """
        处理音频文件
        Args:
            data (dict): JSON数据
            record_id (str): 记录ID
            download_audio (bool): 是否下载音频文件
        Returns:
            list: 音频文件信息列表
        """
        self.log("开始处理音频文件...")
        audio_files = []
        
        if 'data' not in data or 'fileList' not in data['data']:
            self.log("警告：未找到音频文件列表")
            return audio_files
        
        file_list = data['data']['fileList']
        self.log(f"找到 {len(file_list)} 个音频文件")
        
        # 创建以recordId命名的子目录
        record_dir = os.path.join(self.temp_base_dir, str(record_id))
        os.makedirs(record_dir, exist_ok=True)
        self.log(f"✓ 创建记录目录: {record_dir}")
        
        for i, file_info in enumerate(file_list):
            title = file_info.get('title', f'audio_{i}')
            file_url = file_info.get('file_address', '')
            
            if not file_url:
                self.log(f"警告：音频文件 '{title}' 没有有效的URL")
                continue
            
            # 从URL中提取文件扩展名
            parsed_url = urlparse(file_url)
            file_extension = os.path.splitext(parsed_url.path)[1] or '.wav'
            
            # 使用title作为文件名
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_', '（', '）')).rstrip()
            audio_filename = f"{safe_title}{file_extension}"
            audio_path = os.path.join(record_dir, audio_filename)
            
            audio_info = {
                'title': title,
                'filename': audio_filename,
                'local_path': audio_path,
                'original_url': file_url,
                'downloaded': False
            }
            
            if download_audio:
                self.log(f"开始下载音频文件: {title}")
                if self.download_audio_file(file_url, audio_path):
                    self.log(f"✓ 下载成功: {audio_filename}")
                    audio_info['downloaded'] = True
                else:
                    self.log(f"✗ 下载失败: {title}")
            
            audio_files.append(audio_info)
        
        self.log(f"✓ 音频文件处理完成")
        return audio_files
    
    def save_results(self, record_id, questionnaire_results, audio_files):
        """
        保存处理结果
        Args:
            record_id (str): 记录ID
            questionnaire_results (dict): 问卷结果
            audio_files (list): 音频文件信息
        Returns:
            str: 保存目录路径
        """
        record_dir = os.path.join(self.temp_base_dir, str(record_id))
        os.makedirs(record_dir, exist_ok=True)
        
        # 保存问卷结果
        questionnaire_file = os.path.join(record_dir, "questionnaire_results.json")
        with open(questionnaire_file, 'w', encoding='utf-8') as f:
            json.dump(questionnaire_results, f, ensure_ascii=False, indent=4)
        self.log(f"✓ 问卷结果已保存: {questionnaire_file}")
        
        # 保存音频文件信息
        audio_info_file = os.path.join(record_dir, "audio_files_info.json")
        with open(audio_info_file, 'w', encoding='utf-8') as f:
            json.dump(audio_files, f, ensure_ascii=False, indent=4)
        self.log(f"✓ 音频文件信息已保存: {audio_info_file}")
        
        # 保存处理日志
        log_file = os.path.join(record_dir, "processing_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.logs))
        self.log(f"✓ 处理日志已保存: {log_file}")
        
        return record_dir
    
    def process_single_person_data(self, json_file_path, download_audio=True):
        """
        处理单个人的数据：提取问卷结果和下载音频文件
        Args:
            json_file_path (str): JSON数据文件路径
            download_audio (bool): 是否下载音频文件
        Returns:
            dict or None: 处理结果，失败时返回None
        """
        self.log("=" * 60)
        self.log("开始处理个人数据")
        self.log(f"数据文件: {json_file_path}")
        self.log(f"下载音频: {'是' if download_audio else '否'}")
        self.log("=" * 60)
        
        # 1. 加载JSON数据
        data = self.load_json_data(json_file_path)
        if data is None:
            return None
        
        # 2. 获取recordId
        record_id = None
        if 'data' in data and 'recordId' in data['data']:
            record_id = data['data']['recordId']
            self.log(f"✓ 获取到recordId: {record_id}")
        else:
            self.log("警告：未找到recordId，将使用默认值")
            record_id = "unknown"
        
        # 3. 提取问卷结果
        questionnaire_results = self.extract_questionnaire_results(data)
        
        # 4. 处理音频文件
        audio_files = self.process_audio_files(data, record_id, download_audio)
        
        # 5. 保存结果
        record_dir = self.save_results(record_id, questionnaire_results, audio_files)
        
        # 6. 构建结果数据
        result_data = {
            'recordId': record_id,
            'questionnaire_results': questionnaire_results,
            'audio_files': audio_files,
            'record_directory': record_dir,
            'processing_logs': self.logs.copy()
        }
        
        self.log("=" * 60)
        self.log("处理完成")
        self.log(f"RecordId: {record_id}")
        self.log(f"问卷结果数量: {len(questionnaire_results)}")
        self.log(f"音频文件数量: {len(audio_files)}")
        downloaded_count = sum(1 for audio in audio_files if audio.get('downloaded', False))
        self.log(f"已下载音频: {downloaded_count}/{len(audio_files)}")
        self.log(f"数据目录: {record_dir}")
        self.log("=" * 60)
        
        return result_data

def main():
    """主函数"""
    print("=" * 80)
    print("朔黄工程数据处理工具")
    print("功能：从JSON数据中提取问卷结果和音频文件信息")
    print("=" * 80)
    
    # 配置参数
    json_file_path = "E:\\200_main_business\\朔黄工程\\第二阶段数据\\data\\测评包数据结构_2025-06-10_14-20-03.txt"
    temp_directory = "./temp_data"
    download_audio = True  # 设置为False可以只提取信息不下载音频
    
    # 创建数据处理器
    processor = DataProcessor(temp_directory)
    
    # 处理数据
    result = processor.process_single_person_data(json_file_path, download_audio)
    
    if result:
        print("\n" + "=" * 50)
        print("处理结果摘要")
        print("=" * 50)
        print(f"RecordId: {result['recordId']}")
        print(f"问卷结果数量: {len(result['questionnaire_results'])}")
        print(f"音频文件数量: {len(result['audio_files'])}")
        downloaded_count = sum(1 for audio in result['audio_files'] if audio.get('downloaded', False))
        print(f"已下载音频: {downloaded_count}/{len(result['audio_files'])}")
        print(f"数据目录: {result['record_directory']}")
        
        print("\n问卷结果概览:")
        for key, value in result['questionnaire_results'].items():
            if isinstance(value, dict) and 'score' in value:
                print(f"  - {value.get('questionnaireName', key)}: {value['score']} ({value.get('level', 'N/A')})")
            elif key == 'personInfo':
                print(f"  - 个人信息: 已提取")
        
        print("\n音频文件列表:")
        for audio in result['audio_files']:
            status = "✓" if audio.get('downloaded', False) else "✗"
            print(f"  {status} {audio['title']}: {audio['filename']}")
        
        print(f"\n详细信息请查看目录: {result['record_directory']}")
    else:
        print("\n处理失败，请检查日志信息")

if __name__ == "__main__":
    main()