#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整心理评估报告生成器
集成情绪分析和HTML报告生成功能
"""

import json
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import subprocess
from pathlib import Path
import warnings
import requests
from urllib.parse import urlparse
import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)

def create_emotion_spectrum_chart(emotion_scores, user_name):
    """创建情绪谱图"""
    # 指定的情绪顺序（使用中文标签，与JSON文件匹配）
    emotion_order = ['快乐', '愤怒', '悲伤', '中性', '恐惧', '厌恶', '惊讶']
    
    # 检查情绪数据是否为空或无效
    if not emotion_scores or all(score == 0 or score is None for score in emotion_scores.values()):
        # 如果情绪数据为空或全为0，创建全0的图表
        ordered_emotions = emotion_order
        ordered_scores = [0] * len(emotion_order)
        max_score_index = 0  # 默认第一个为橙色
    else:
        # 按指定顺序获取情绪数据
        ordered_emotions = []
        ordered_scores = []
        
        for emotion in emotion_order:
            if emotion in emotion_scores:
                ordered_emotions.append(emotion)
                ordered_scores.append(emotion_scores[emotion])  # 直接使用分数，不需要转换
        
        # 找到最高分数的情绪
        max_score_index = np.argmax(ordered_scores) if ordered_scores else 0
    
    # 设置颜色：最高分为橙色，其他为深灰色
    colors = ['#4a4a4a'] * len(ordered_scores)
    if ordered_scores:  # 只有当有分数时才设置橙色
        colors[max_score_index] = '#ff8c00'  # 橙色
    
    # 创建水平条形图
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=ordered_emotions,
        x=ordered_scores,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'{score:.1f}' for score in ordered_scores],
        textposition='outside',
        textfont=dict(size=12, color='white'),
        hovertemplate='%{y}: %{x:.1f}<extra></extra>'
    ))
    
    # 设置布局
    fig.update_layout(
        title='',  # 删除标题
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, max(ordered_scores) * 1.15 if ordered_scores else 30]
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color='white'),
            categoryorder='array',
            categoryarray=ordered_emotions[::-1]  # 反转顺序使第一个在顶部
        ),
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d',
        font=dict(color='white'),
        margin=dict(l=60, r=180, t=20, b=20),
        height=300,  # 调整为细长样式
        width=600,
        showlegend=False
    )
    
    return fig

def load_template_data():
    """加载心理评估报告模板数据"""
    # 尝试多个可能的路径
    possible_paths = [
        '心理评估报告模板.json',
        os.path.join(os.path.dirname(__file__), '心理评估报告模板.json'),
        '../心理评估报告模板.json'
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            continue
    
    # 如果所有路径都失败，抛出详细错误
    raise FileNotFoundError(f"无法找到心理评估报告模板.json文件。尝试的路径: {possible_paths}")

def get_ghq12_assessment(ghq12_score, template_data):
    """根据GHQ12总分获取心理风险评估文本"""
    if ghq12_score == 0:
        return template_data['心理风险评估']['GHQ12评估']['无风险']
    elif 1 <= ghq12_score <= 3:
        return template_data['心理风险评估']['GHQ12评估']['低风险']
    else:  # >= 4
        return template_data['心理风险评估']['GHQ12评估']['高风险']

def get_perma_assessment(perma_score, template_data):
    """根据PERMA总分获取幸福感评估文本"""
    if perma_score >= 80:
        return template_data['心理风险评估']['幸福感评估']['高幸福感']
    elif 40 <= perma_score <= 79:
        return template_data['心理风险评估']['幸福感评估']['中幸福感']
    else:  # < 40
        return template_data['心理风险评估']['幸福感评估']['低幸福感']

def get_dimension_level(score, total_scores):
    """根据分数和总体分布确定维度等级"""
    # 将用户分数加入总体分数列表并排序
    all_scores = total_scores + [score]
    sorted_scores = sorted(all_scores, reverse=True)
    total_count = len(sorted_scores)
    
    # 找到用户分数的排名
    rank = sorted_scores.index(score) + 1
    
    # 计算阈值
    high_threshold = int(total_count * 0.35)
    low_threshold = int(total_count * 0.9)
    
    if rank <= high_threshold:
        return '高分'
    elif rank <= low_threshold:
        return '中分'
    else:
        return '低分'

def get_dimension_analysis_text(psychology_results, job_type, template_data):
    """获取三大维度分析结构（用于JSON）"""
    # 检查三维度结果是否为空或无效
    if not psychology_results or all(value == 0 or value is None for value in psychology_results.values()):
        # 如果三维度数据为空，返回默认结构
        return {
            '个性画像': '您的语音输入未输入或不清晰，无法呈现此部分结果。',
            '支持系统': '您的语音输入未输入或不清晰，无法呈现此部分结果。',
            '职场感知': '您的语音输入未输入或不清晰，无法呈现此部分结果。'
        }
    
    # 模拟总体分数分布（实际应用中应从数据库获取）
    total_scores = [60, 70, 80, 50, 90, 40, 30, 75, 65, 85]
    
    dimensions = ['个性画像', '支持系统', '职场感知']
    dimension_keys = ['个性画像', '支持系统', '职场感知']
    
    analysis_result = {}
    
    for i, dim in enumerate(dimensions):
        score = psychology_results.get(dimension_keys[i], 0)
        
        # 检查单个维度是否为空或0
        if score == 0 or score is None:
            analysis_result[dim] = '您的语音输入未输入或不清晰，无法呈现此部分结果。'
            continue
            
        level = get_dimension_level(score, total_scores)
        
        # 根据工种选择对应的文本
        if job_type == '普通职工':
            job_key = '基层员工'
        elif job_type == '中层管理':
            job_key = '中层员工'
        else:
            job_key = '高层员工'
        
        dim_text = template_data['三大维度分析'][dim][job_key][level]
        analysis_result[dim] = dim_text
    
    return analysis_result

# 导入情绪分析模块
from predict_emotion2vec import analyze_emotion_for_user, predict_emotion_from_audio
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，无法生成可视化图表")

# PDF生成相关导入
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("警告：pdfkit未安装，PDF生成功能将不可用")

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    WEASYPRINT_AVAILABLE = False
    if "pango" in str(e).lower():
        print("警告：WeasyPrint无法加载Pango库，将使用pdfkit作为替代方案")
    else:
        print(f"警告：weasyprint导入失败: {e}")
    print("将尝试使用pdfkit生成PDF")



def find_wkhtmltopdf():
    """查找wkhtmltopdf可执行文件路径"""
    import shutil
    
    # 常见的安装路径
    common_paths = [
        r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
        r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
        r'C:\wkhtmltopdf\bin\wkhtmltopdf.exe',
    ]
    
    # 首先检查PATH中是否有wkhtmltopdf
    if shutil.which('wkhtmltopdf'):
        return shutil.which('wkhtmltopdf')
    
    # 检查常见安装路径
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))

def generate_pdf_from_html(html_path, output_dir=None):
    """
    将HTML文件转换为PDF文件
    
    Args:
        html_path (str): HTML文件路径
        output_dir (str, optional): 输出目录，默认为None
    
    Returns:
        str: 生成的PDF文件路径，失败时返回None
    """
    if not os.path.exists(html_path):
        print(f"错误：HTML文件不存在: {html_path}")
        return None
    
    # 生成PDF文件路径
    pdf_path = html_path.replace('.html', '.pdf')
    
    # 尝试使用weasyprint生成PDF
    if WEASYPRINT_AVAILABLE:
        try:
            HTML(filename=html_path).write_pdf(pdf_path)
            print(f"使用weasyprint生成PDF成功: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"weasyprint生成PDF失败: {e}")
    
    # 尝试使用pdfkit生成PDF
    if PDFKIT_AVAILABLE:
        try:
            # 查找wkhtmltopdf路径
            wkhtmltopdf_path = find_wkhtmltopdf()
            
            if not wkhtmltopdf_path:
                print("错误：未找到wkhtmltopdf可执行文件")
                print("请从 https://wkhtmltopdf.org/downloads.html 下载并安装wkhtmltopdf")
            else:
                print(f"找到wkhtmltopdf: {wkhtmltopdf_path}")
                
                # 配置pdfkit选项
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                    'no-outline': None,
                    'enable-local-file-access': None,
                    'disable-smart-shrinking': None,
                    'print-media-type': None
                }
                
                # 配置wkhtmltopdf路径
                config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
                
                # 生成PDF
                pdfkit.from_file(html_path, pdf_path, options=options, configuration=config)
                print(f"使用pdfkit生成PDF成功: {pdf_path}")
                return pdf_path
                
        except Exception as e:
            print(f"pdfkit生成PDF失败: {e}")
            if "No wkhtmltopdf executable found" in str(e):
                print("解决方案：")
                print("1. 从 https://wkhtmltopdf.org/downloads.html 下载wkhtmltopdf")
                print("2. 安装到默认路径或添加到系统PATH")
                print("3. 重新运行脚本")
    
    print("警告：无法生成PDF文件，请安装weasyprint或pdfkit库")
    return None



def generate_emotion_visualization(json_file_path):
    """
    根据情绪分析结果生成可视化折线图
    
    Args:
        json_file_path (str): 情绪分析结果JSON文件路径
    
    Returns:
        str: 生成的图片文件路径，失败时返回None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("错误：matplotlib未安装，无法生成可视化图表")
        return None
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否有按题目分组的结果
        if '按题目分组结果' not in data or not data['按题目分组结果']:
            print("警告：未找到按题目分组的情绪分析结果，无法生成可视化图表")
            return None
        
        grouped_results = data['按题目分组结果']
        
        # 获取所有情绪类型
        all_emotions = set()
        for topic_results in grouped_results.values():
            if topic_results:
                all_emotions.update(topic_results.keys())
        
        if not all_emotions:
            print("警告：没有有效的情绪数据，无法生成可视化图表")
            return None
        
        # 按情绪得分排序情绪类型，确保图例顺序一致
        emotion_order = sorted(all_emotions)
        
        # 准备数据
        topics = list(grouped_results.keys())
        topics.sort()  # 按题目名称排序
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 为每种情绪绘制折线
        colors = plt.cm.tab10(range(len(emotion_order)))
        
        for i, emotion in enumerate(emotion_order):
            scores = []
            valid_topics = []
            
            for topic in topics:
                if topic in grouped_results and grouped_results[topic]:
                    score = grouped_results[topic].get(emotion, 0)
                    scores.append(score)
                    valid_topics.append(topic)
            
            if scores:  # 只有当有数据时才绘制
                plt.plot(range(len(valid_topics)), scores, 
                        marker='o', linewidth=2, markersize=6,
                        label=emotion, color=colors[i])
        
        # 设置图表属性
        plt.title('各题目情绪分析结果对比', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('题目', fontsize=12)
        plt.ylabel('情绪得分', fontsize=12)
        
        # 设置x轴标签
        if valid_topics:
            plt.xticks(range(len(valid_topics)), valid_topics, rotation=45, ha='right')
        
        # 设置y轴范围
        plt.ylim(0, 100)
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_dir = os.path.dirname(json_file_path)
        user_id = os.path.basename(json_file_path).replace('_结果.json', '')
        image_path = os.path.join(output_dir, f"{user_id}_情绪分析折线图.png")
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"情绪分析可视化图表已生成: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"生成可视化图表时出现错误: {e}")
        return None

def load_json_data_direct(json_file_path):
    """
    直接从JSON文件加载数据
    Args:
        json_file_path (str): JSON文件路径
    Returns:
        dict or None: 加载的数据，失败时返回None
    """
    print(f"开始加载JSON数据文件: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        print(f"错误：JSON文件不存在 - {json_file_path}")
        return None
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ JSON文件加载成功")
        return data
    except Exception as e:
        print(f"错误：加载JSON文件失败 - {e}")
        return None

def extract_questionnaire_results_direct(data):
    """
    从JSON数据中直接提取问卷结果
    Args:
        data (dict): JSON数据
    Returns:
        dict: 提取的问卷结果
    """
    print("开始提取问卷结果...")
    questionnaire_results = {}
    
    if 'data' not in data or 'dataList' not in data['data']:
        print("警告：未找到问卷数据列表")
        return questionnaire_results
    
    data_list = data['data']['dataList']
    print(f"找到 {len(data_list)} 项问卷数据")
    
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
        
        print(f"  - {questionnaire_name}: {score} ({level})")
    
    # 添加个人信息
    if 'data' in data and 'personInfo' in data['data']:
        person_info = data['data']['personInfo']
        questionnaire_results['personInfo'] = person_info
        print(f"✓ 个人信息已添加")
    
    print(f"✓ 问卷结果提取完成，共 {len(questionnaire_results)} 项")
    return questionnaire_results

def download_audio_file_direct(url, save_path, timeout=30):
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
        print(f"下载失败: {e}")
        return False

def process_audio_files_direct(data, record_id, download_audio=True, temp_base_dir="./temp_data"):
    """
    直接处理音频文件
    Args:
        data (dict): JSON数据
        record_id (str): 记录ID
        download_audio (bool): 是否下载音频文件
        temp_base_dir (str): 临时目录基路径
    Returns:
        list: 音频文件信息列表
    """
    print("开始处理音频文件...")
    audio_files = []
    
    if 'data' not in data or 'fileList' not in data['data']:
        print("警告：未找到音频文件列表")
        return audio_files
    
    file_list = data['data']['fileList']
    print(f"找到 {len(file_list)} 个音频文件")
    
    # 创建以recordId命名的子目录
    record_dir = os.path.join(temp_base_dir, str(record_id))
    os.makedirs(record_dir, exist_ok=True)
    print(f"✓ 创建记录目录: {record_dir}")
    
    for i, file_info in enumerate(file_list):
        title = file_info.get('title', f'audio_{i}')
        file_url = file_info.get('file_address', '')
        
        # 检查是否为预热题，如果是则跳过下载
        if '预热题' in title:
            print(f"跳过预热题音频: {title}")
            continue
        
        if not file_url:
            print(f"警告：音频文件 '{title}' 没有有效的URL")
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
            print(f"开始下载音频文件: {title}")
            if download_audio_file_direct(file_url, audio_path):
                print(f"✓ 下载成功: {audio_filename}")
                audio_info['downloaded'] = True
            else:
                print(f"✗ 下载失败: {title}")
        
        audio_files.append(audio_info)
    
    print(f"✓ 音频文件处理完成")
    return audio_files

def extract_emotion_data_from_results(emotion_results):
    """从内存中的情绪分析结果提取情绪数据"""
    if not emotion_results:
        return {}
    
    # 获取融合情绪结果中的各项情绪得分
    emotion_scores = emotion_results.get('融合情绪结果', {}).get('各项情绪得分', {})
    return emotion_scores

def load_user_data():
    """加载用户综合报告数据"""
    # 尝试多个可能的路径
    possible_paths = [
        'temp_data/用户综合报告.json',
        os.path.join(os.path.dirname(__file__), 'temp_data/用户综合报告.json'),
        '../temp_data/用户综合报告.json'
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            continue
    
    # 如果所有路径都失败，抛出详细错误
    raise FileNotFoundError(f"无法找到用户综合报告.json文件。尝试的路径: {possible_paths}")

def get_dimension_analysis(dimension_results, job_type, template_data):
    """生成多维度分析内容"""
    if not dimension_results:
        return "<p>暂无多维度分析数据。</p>"
    
    analysis_html = ""
    for dimension, score in dimension_results.items():
        if score is not None:
            analysis_html += f"""
            <div class="assessment-item">
                <h4>{dimension}</h4>
                <p>得分：{score:.2f}</p>
            </div>
            """
    
    return analysis_html if analysis_html else "<p>暂无有效的多维度分析数据。</p>"

def generate_html_report_from_data(user_id, emotion_results, output_dir="results", pdf_optimized=False, user_data_dict=None):
    """
    从内存中的情绪分析结果直接生成HTML报告
    
    Args:
        user_id (str): 用户ID
        emotion_results (dict): 情绪分析结果字典
        output_dir (str): 输出目录
        pdf_optimized (bool): 是否为PDF优化布局（连续长页面）
        user_data_dict (dict): 用户数据字典，如果提供则使用此数据而不是从文件加载
    
    Returns:
        str: 生成的HTML文件路径，失败时返回None
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 从内存数据中提取情绪得分
        emotion_scores = extract_emotion_data_from_results(emotion_results)
        
        # 创建情绪谱图
        emotion_chart = create_emotion_spectrum_chart(emotion_scores, user_id)
        emotion_chart_html = pyo.plot(emotion_chart, output_type='div', include_plotlyjs=True)
        
        # 加载模板数据
        template_data = load_template_data()
        
        # 获取用户数据
        if user_data_dict is not None:
            # 使用传入的用户数据
            user_info = user_data_dict
            ghq12_score = user_info.get('GHQ12总分', 0)
            perma_score = user_info.get('PERMA总分', 50)
            job_type = user_info.get('工种', '普通职工')
        else:
            # 尝试从文件加载用户数据
            try:
                user_data = load_user_data()
                if str(user_id) in user_data:
                    user_info = user_data[str(user_id)]['问卷数据']
                elif user_id in user_data:
                    user_info = user_data[user_id]['问卷数据']
                else:
                    # 如果找不到用户数据，使用默认值
                    print(f"警告：未找到用户 {user_id} 的数据，使用默认值")
                    user_info = {'GHQ12总分': 0, 'PERMA总分': 50, '工种': '普通职工'}
                ghq12_score = user_info.get('GHQ12总分', 0)
                perma_score = user_info.get('PERMA总分', 50)
                job_type = user_info.get('工种', '普通职工')
            except Exception as e:
                print(f"警告：加载用户数据失败 {e}，使用默认值")
                ghq12_score = 0
                perma_score = 50
                job_type = '普通职工'
        
        ghq12_text = get_ghq12_assessment(ghq12_score, template_data)
        perma_text = get_perma_assessment(perma_score, template_data)
        
        # 生成情绪描述
        if emotion_scores and any(score > 0 for score in emotion_scores.values() if score is not None):
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1] if x[1] is not None else 0)
            emotion_description = f"根据语音情绪分析，您的主要情绪倾向为{max_emotion[0]}（得分：{max_emotion[1]:.1f}）。"
        else:
            emotion_description = "暂无有效的情绪分析数据。"
        
        # 生成多维度分析
        dimension_analysis = get_dimension_analysis(emotion_results.get('三维度结果', {}), job_type, template_data)
        
        # 获取当前时间
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        # 根据是否为PDF优化选择不同的样式
        if pdf_optimized:
            # PDF优化样式：连续长页面，去除分页断开
            body_style = """
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            line-height: 1.8;
            margin: 0;
            padding: 40px 60px;
            background-color: #ffffff;
            color: #333333;
            font-size: 14px;
            """
            container_style = """
            max-width: none;
            margin: 0;
            background-color: #ffffff;
            padding: 0;
            border-radius: 0;
            box-shadow: none;
            """
            section_style = """
            margin: 40px 0;
            padding: 25px 0;
            background-color: #ffffff;
            border-radius: 0;
            border-left: 4px solid #2c5aa0;
            border-bottom: 1px solid #e0e0e0;
            page-break-inside: avoid;
            """
            header_style = """
            text-align: center;
            margin-bottom: 50px;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 30px;
            page-break-after: avoid;
            """
            h1_color = "#2c5aa0"
            h2_color = "#2c5aa0"
            h3_color = "#4a90e2"
            h4_color = "#2c5aa0"
            assessment_bg = "#f8f9fa"
            chart_bg = "#ffffff"
        else:
            # 原始暗色主题样式
            body_style = """
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background-color: #1e1e1e;
            color: #ffffff;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            """
            container_style = """
            max-width: 1000px;
            margin: 0 auto;
            background-color: #2d2d2d;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            """
            section_style = """
            margin: 30px 0;
            padding: 20px;
            background-color: #3a3a3a;
            border-radius: 8px;
            border-left: 4px solid #ff8c00;
            """
            header_style = """
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #ff8c00;
            padding-bottom: 20px;
            """
            h1_color = "#ff8c00"
            h2_color = "#ff8c00"
            h3_color = "#ffa500"
            h4_color = "#ff8c00"
            assessment_bg = "#4a4a4a"
            chart_bg = "#2d2d2d"
        
        # HTML模板
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>心理评估报告 - {user_id}</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm 1.5cm;
        }}
        body {{
            {body_style}
        }}
        .container {{
            {container_style}
        }}
        .header {{
            {header_style}
        }}
        .header h1 {{
            color: {h1_color};
            margin: 0;
            font-size: {'32px' if pdf_optimized else '28px'};
            font-weight: bold;
        }}
        .header p {{
            color: {'#666666' if pdf_optimized else '#cccccc'};
            margin: 15px 0 0 0;
            font-size: {'16px' if pdf_optimized else '14px'};
        }}
        .section {{
            {section_style}
        }}
        .section h2 {{
            color: {h2_color};
            margin-top: 0;
            font-size: {'24px' if pdf_optimized else '20px'};
            font-weight: bold;
            margin-bottom: 20px;
        }}
        .section h3 {{
            color: {h3_color};
            margin-top: 20px;
            font-size: {'18px' if pdf_optimized else '16px'};
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .assessment-item {{
            margin: 20px 0;
            padding: {'20px' if pdf_optimized else '15px'};
            background-color: {assessment_bg};
            border-radius: {'0' if pdf_optimized else '5px'};
            {'border: 1px solid #e0e0e0;' if pdf_optimized else ''}
        }}
        .assessment-item h4 {{
            color: {h4_color};
            margin: 0 0 12px 0;
            font-size: {'16px' if pdf_optimized else '14px'};
            font-weight: bold;
        }}
        .assessment-item p {{
            margin: 8px 0;
            line-height: {'1.8' if pdf_optimized else '1.6'};
            font-size: {'14px' if pdf_optimized else 'inherit'};
        }}
        .chart-container {{
            background-color: {chart_bg};
            padding: {'30px 20px' if pdf_optimized else '20px'};
            border-radius: {'0' if pdf_optimized else '8px'};
            margin: {'30px 0' if pdf_optimized else '20px 0'};
            {'border: 1px solid #e0e0e0;' if pdf_optimized else ''}
            text-align: center;
        }}
        .chart-description {{
            margin-top: 20px;
            padding: 15px;
            background-color: {'#f0f8ff' if pdf_optimized else '#3a3a3a'};
            border-radius: {'0' if pdf_optimized else '5px'};
            color: {'#333333' if pdf_optimized else '#ffffff'};
            font-size: {'14px' if pdf_optimized else 'inherit'};
            line-height: 1.6;
            {'border-left: 4px solid #2c5aa0;' if pdf_optimized else ''}
        }}
        .footer {{
            text-align: center;
            margin-top: {'60px' if pdf_optimized else '40px'};
            padding-top: {'30px' if pdf_optimized else '20px'};
            border-top: 1px solid {'#cccccc' if pdf_optimized else '#555'};
            color: {'#888888' if pdf_optimized else '#888'};
            font-size: {'12px' if pdf_optimized else '12px'};
            page-break-inside: avoid;
        }}
        {'@media print { .section { page-break-inside: avoid; } .assessment-item { page-break-inside: avoid; } }' if pdf_optimized else ''}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>心理评估报告</h1>
            <p>生成时间：{current_time}</p>
        </div>
        
        <div class="section">
            <h2>一、整体情况</h2>
            
            <h3>1. 心理风险评估</h3>
            <div class="assessment-item">
                <h4>GHQ12评估</h4>
                <p>{ghq12_text}</p>
            </div>
            
            <h3>2. 幸福感评估</h3>
            <div class="assessment-item">
                <h4>PERMA评估</h4>
                <p>{perma_text}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>二、情绪谱图分析</h2>
            <div class="chart-container">
                {emotion_chart_html}
            </div>
            <p>{emotion_description}</p>
        </div>
        
        <div class="section">
            <h2>三、多维度分析</h2>
            {dimension_analysis}
        </div>
        
        <div class="footer">
            <p>本测评基于科学的评估体系，用于反映个体心理状态的大致情况，不具备临床诊断效力。</p>
            <p>若您对测评结果持有疑虑，建议及时前往正规医疗机构进行专业诊疗。</p>
        </div>
    </div>
</body>
</html>
        """
        
        # 生成HTML文件
        html_file_path = os.path.join(output_dir, f"{user_id}_心理评估报告.html")
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"HTML报告生成成功: {html_file_path}")
        return html_file_path
        
    except Exception as e:
        print(f"生成HTML报告时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_complete_report_from_json(json_file_path, **kwargs):
    """
    直接从JSON文件生成完整的心理评估报告（集成版本）
    Args:
        json_file_path (str): JSON数据文件路径
        **kwargs: 其他参数
    Returns:
        str or None: 生成的报告文件路径，失败时返回None
    """
    print("=" * 60)
    print("开始从JSON文件生成完整报告（集成版本）")
    print(f"数据文件: {json_file_path}")
    print("=" * 60)
    
    # 1. 直接加载JSON数据
    data = load_json_data_direct(json_file_path)
    if data is None:
        return None
    
    # 2. 获取recordId
    record_id = None
    if 'data' in data and 'recordId' in data['data']:
        record_id = data['data']['recordId']
        print(f"✓ 获取到recordId: {record_id}")
    else:
        print("警告：未找到recordId，将使用默认值")
        record_id = "unknown"
    
    # 3. 直接提取问卷结果
    questionnaire_data = extract_questionnaire_results_direct(data)
    
    # 4. 处理音频文件（可选择是否下载）
    download_audio = kwargs.get('download_audio', True)
    temp_base_dir = kwargs.get('temp_base_dir', './temp_data')
    audio_files_info = process_audio_files_direct(data, record_id, download_audio, temp_base_dir)
    
    # 5. 获取音频文件路径列表用于情绪分析
    record_dir = os.path.join(temp_base_dir, str(record_id))
    audio_files = [info['local_path'] for info in audio_files_info if info.get('downloaded', False)]
    
    if not audio_files:
        print(f"警告：在目录 {record_dir} 中未找到音频文件")
    else:
        print(f"找到 {len(audio_files)} 个音频文件")
    
    # 6. 进行语音情绪分析
    emotion_results = {}
    if audio_files:
        try:
            print("开始语音情绪分析...")
            
            # 准备参数
            model_path = kwargs.get('model_path')
            device = kwargs.get('device', 'cpu')
            exclude_labels = kwargs.get('exclude_labels', ['预热题'])
            include_labels = kwargs.get('include_labels')
            
            # 批量处理音频文件进行情绪分析
            from predict_emotion2vec import predict_emotions_batch
            
            # 过滤音频文件
            valid_audio_files = []
            valid_labels = []
            
            for audio_file in audio_files:
                # 从文件名提取标签（'-'之前的部分）
                filename = os.path.basename(audio_file)
                label = filename.split('-')[0] if '-' in filename else filename.replace('.wav', '')
                
                # 检查是否需要排除或包含此标签
                if exclude_labels and label in exclude_labels:
                    print(f"跳过排除的标签: {label}")
                    continue
                
                if include_labels and label not in include_labels:
                    print(f"跳过未包含的标签: {label}")
                    continue
                
                valid_audio_files.append(audio_file)
                valid_labels.append(label)
            
            print(f"开始批量分析 {len(valid_audio_files)} 个音频文件...")
            
            # 批量进行情绪分析
            try:
                batch_results = predict_emotions_batch(
                    valid_audio_files,
                    model_path=model_path,
                    device=device
                )
                
                audio_emotion_results = {}
                for i, (audio_file, label) in enumerate(zip(valid_audio_files, valid_labels)):
                    filename = os.path.basename(audio_file)
                    if i < len(batch_results) and batch_results[i] is not None:
                        audio_emotion_results[label] = batch_results[i]
                        print(f"  - {label}: 分析完成")
                    else:
                        print(f"  - {label}: 分析失败或静音")
                        
            except Exception as e:
                print(f"批量分析出错: {e}")
                # 如果批量处理失败，回退到单个处理
                print("回退到单个文件处理模式...")
                from predict_emotion2vec import predict_emotion_from_audio
                
                audio_emotion_results = {}
                for audio_file, label in zip(valid_audio_files, valid_labels):
                    filename = os.path.basename(audio_file)
                    print(f"分析音频文件: {filename} (标签: {label})")
                    
                    try:
                        result = predict_emotion_from_audio(
                            audio_file, 
                            model_path=model_path, 
                            device=device
                        )
                        
                        if result:
                            audio_emotion_results[label] = result
                            print(f"  - {label}: 分析完成")
                        else:
                            print(f"  - {label}: 分析失败")
                            
                    except Exception as e:
                        print(f"  - {label}: 分析出错 - {e}")
            
            # 计算总体统计
            if audio_emotion_results:
                all_emotions = {}
                has_valid_audio = False  # 标记是否有有效的音频文件
                
                for label_results in audio_emotion_results.values():
                    if isinstance(label_results, dict):
                        for emotion, score in label_results.items():
                            if emotion not in all_emotions:
                                all_emotions[emotion] = []
                            all_emotions[emotion].append(score)
                            if score > 0:
                                has_valid_audio = True
                
                # 计算平均值
                average_emotions = {
                    emotion: round(sum(scores) / len(scores), 2) 
                    for emotion, scores in all_emotions.items()
                }
                
                # 找到得分最高的情绪作为主要情绪标签
                if has_valid_audio and any(score > 0 for score in average_emotions.values()):
                    main_emotion = max(average_emotions.items(), key=lambda x: x[1])[0]
                else:
                    main_emotion = '无'  # 如果所有文件都是静音或情绪得分都为0，则标记为'无'
                
                # 为HTML报告生成器添加融合情绪结果字段
                emotion_results['融合情绪结果'] = {
                    '情绪标签': main_emotion,
                    '各项情绪得分': average_emotions
                }
            else:
                # 如果是静音文件，情绪值设为0
                default_emotions = {
                    '中性': 0,
                    '快乐': 0,
                    '恐惧': 0,
                    '悲伤': 0,
                    '惊讶': 0,
                    '愤怒': 0,
                    '平静': 0,
                    '厌恶': 0
                }
                # 为HTML报告生成器添加融合情绪结果字段
                emotion_results['融合情绪结果'] = {
                    '情绪标签': '暂无有效的情绪分析数据。',
                    '各项情绪得分': default_emotions
                }
            
            print("语音情绪分析完成")
            
            # 计算三维度结果
            try:
                print("开始计算三维度结果...")
                from predict_emotion2vec import calculate_scale_results_by_labels
                
                if audio_emotion_results:
                    scale_results = calculate_scale_results_by_labels(audio_emotion_results)
                    emotion_results['三维度结果'] = scale_results
                    print("三维度结果计算完成")
                else:
                    print("没有有效的语音情绪数据，跳过三维度计算")
                    emotion_results['三维度结果'] = {}
                    
            except Exception as e:
                print(f"计算三维度结果时出现异常: {e}")
                emotion_results['三维度结果'] = {}
            
        except Exception as e:
            print(f"执行语音情绪分析时出现异常: {e}")
            emotion_results = {}
    
    print("开始生成最终报告...")
    
    # 提取person_info，供后续使用
    person_info = questionnaire_data.get('personInfo', {})
    
    # 7. 生成HTML报告
    html_path = None
    try:
        # 使用本地函数，无需导入
        
        # 构造用于HTML报告的数据结构
        report_data = {
            'user_id': record_id,
            'questionnaire_results': questionnaire_data,
            'emotion_results': emotion_results
        }
        
        # 转换问卷数据格式以匹配HTML报告生成器的期望格式
        user_info = {
            'GHQ12总分': questionnaire_data.get('一般健康问卷（GHQ-12）_一般健康问卷（GHQ-12）>总分', {}).get('score', 0),
            'PERMA总分': questionnaire_data.get('PERMA幸福量表_PERMA幸福量表>总分', {}).get('score', 0),
            '工种': person_info.get('工种', '普通职工')
        }
        
        html_path = generate_html_report_from_data(record_id, emotion_results, record_dir, False, user_info)
        
        if html_path and os.path.exists(html_path):
            print(f"HTML报告生成成功: {html_path}")
        else:
            print("HTML报告生成失败")
            
    except ImportError as e:
        print(f"错误：无法导入HTML报告生成模块: {e}")
    except Exception as e:
        print(f"生成HTML报告时出现异常: {e}")
    
    # 8. 生成最终的JSON报告
    try:
        output_dir = kwargs.get('output_dir', record_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载心理评估报告模板
        template_data = load_template_data()
        
        # 获取心理风险评估和幸福感评估描述
        ghq12_score = questionnaire_data.get('一般健康问卷（GHQ-12）_一般健康问卷（GHQ-12）>总分', {}).get('score', 0)
        perma_score = questionnaire_data.get('PERMA幸福量表_PERMA幸福量表>总分', {}).get('score', 0)
        
        psychological_risk_assessment = get_ghq12_assessment(ghq12_score, template_data)
        happiness_assessment = get_perma_assessment(perma_score, template_data)
        
        # 获取多维度分析描述
        job_type = person_info.get('工种', '普通职工')
        psychology_results = emotion_results.get('三维度结果', {})
        multidimensional_analysis = get_dimension_analysis_text(psychology_results, job_type, template_data)
        
        # 构造最终报告数据
        final_report = {
            'personInfo': person_info,
            'emotion_analysis': {
                '整体情况': {
                    '心理风险评估-GHQ12': psychological_risk_assessment,
                    '幸福感评估-PERMA': happiness_assessment
                },
                '情绪谱图分析': emotion_results.get('融合情绪结果', {}),
                '多维度分析': multidimensional_analysis,
                '备注': '本测评基于科学的评估体系，用于反映个体心理状态的大致情况，不具备临床诊断效力。\n若您对测评结果持有疑虑，建议及时前往正规医疗机构进行专业诊疗。'
            },
            'html_report_path': os.path.abspath(html_path) if html_path and os.path.exists(html_path) else None
        }
        
        # 保存最终报告
        final_report_path = os.path.join(output_dir, f'{record_id}_complete_report.json')
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"完整报告已保存: {final_report_path}")
        
        # 可选：清理中间文件
        cleanup_temp_files = kwargs.get('cleanup_temp_files', False)
        if cleanup_temp_files:
            print("清理中间文件...")
            # 删除音频文件（如果不需要保留）
            for audio_info in audio_files_info:
                if audio_info.get('downloaded', False) and os.path.exists(audio_info['local_path']):
                    try:
                        os.remove(audio_info['local_path'])
                        print(f"✓ 已删除音频文件: {audio_info['filename']}")
                    except Exception as e:
                        print(f"删除音频文件失败: {e}")
        
        print("=" * 60)
        print("集成版本报告生成完成")
        print(f"RecordId: {record_id}")
        print(f"问卷结果数量: {len(questionnaire_data)}")
        print(f"音频文件数量: {len(audio_files_info)}")
        downloaded_count = sum(1 for audio in audio_files_info if audio.get('downloaded', False))
        print(f"已下载音频: {downloaded_count}/{len(audio_files_info)}")
        print(f"最终报告: {final_report_path}")
        print("=" * 60)
        
        return final_report_path
        
    except Exception as e:
        print(f"生成最终报告时出现异常: {e}")
        return None

def generate_complete_report_from_temp_data(record_id, **kwargs):
    """
    从temp_data目录为指定recordId生成完整的心理评估报告（包含情绪分析和HTML报告）
    
    Args:
        record_id (str): 记录ID
        **kwargs: 其他可选参数
            - temp_data_dir: temp_data目录路径
            - model_path: 模型路径
            - output_dir: 输出目录
            - device: 设备类型 (cpu/cuda)
            - exclude_labels: 排除的标签列表
            - include_labels: 包含的标签列表
    
    Returns:
        str: 生成的JSON报告文件路径，失败时返回None
    """
    print(f"开始为记录ID '{record_id}' 生成完整的心理评估报告...")
    
    # 设置temp_data目录路径
    temp_data_dir = kwargs.get('temp_data_dir', 'E:\\200_main_business\\朔黄工程\\第二阶段数据\\code\\语音端到端demo-6\\temp_data')
    record_dir = os.path.join(temp_data_dir, record_id)
    
    # 检查记录目录是否存在
    if not os.path.exists(record_dir):
        print(f"错误：记录目录不存在: {record_dir}")
        return None
    
    # 读取问卷结果
    questionnaire_path = os.path.join(record_dir, 'questionnaire_results.json')
    if not os.path.exists(questionnaire_path):
        print(f"错误：问卷结果文件不存在: {questionnaire_path}")
        return None
    
    try:
        with open(questionnaire_path, 'r', encoding='utf-8') as f:
            questionnaire_data = json.load(f)
        print("问卷结果读取成功")
    except Exception as e:
        print(f"读取问卷结果时出现错误: {e}")
        return None
    
    # 获取音频文件列表
    audio_files = []
    for file in os.listdir(record_dir):
        if file.endswith('.wav'):
            audio_files.append(os.path.join(record_dir, file))
    
    if not audio_files:
        print(f"警告：在目录 {record_dir} 中未找到音频文件")
    else:
        print(f"找到 {len(audio_files)} 个音频文件")
    
    # 第一步：进行语音情绪分析
    emotion_results = {}
    if audio_files:
        try:
            print("开始语音情绪分析...")
            
            # 准备参数
            model_path = kwargs.get('model_path')
            device = kwargs.get('device', 'cpu')
            exclude_labels = kwargs.get('exclude_labels', ['预热题'])
            include_labels = kwargs.get('include_labels')
            
            # 对每个音频文件进行情绪分析
            from predict_emotion2vec import predict_emotion_from_audio
            
            audio_emotion_results = {}
            for audio_file in audio_files:
                # 注意：此函数已废弃，建议使用generate_complete_report函数
                # 从文件名提取标签（'-'之前的部分）
                filename = os.path.basename(audio_file)
                label = filename.split('-')[0] if '-' in filename else filename.replace('.wav', '')
                
                # 检查是否需要排除或包含此标签
                if exclude_labels and label in exclude_labels:
                    print(f"跳过排除的标签: {label}")
                    continue
                
                if include_labels and label not in include_labels:
                    print(f"跳过未包含的标签: {label}")
                    continue
                
                print(f"分析音频文件: {filename} (标签: {label})")
                
                try:
                    # 进行情绪分析
                    result = predict_emotion_from_audio(
                        audio_file, 
                        model_path=model_path, 
                        device=device
                    )
                    
                    if result:
                        audio_emotion_results[label] = result
                        print(f"  - {label}: 分析完成")
                    else:
                        print(f"  - {label}: 分析失败")
                        
                except Exception as e:
                    print(f"  - {label}: 分析出错 - {e}")
            
            emotion_results = {
                '心理风险评估': {},
                '幸福感评估': {},
                '总体统计': {},
                '多维度分析': {
                    '个性画像': {},
                    '支持系统': {},
                    '职场感知': {}
                }
            }
            
            # 计算总体统计
            if audio_emotion_results:
                all_emotions = {}
                has_valid_audio = False  # 标记是否有有效的音频文件
                
                for label_results in audio_emotion_results.values():
                    if isinstance(label_results, dict):
                        for emotion, score in label_results.items():
                            if emotion not in all_emotions:
                                all_emotions[emotion] = []
                            all_emotions[emotion].append(score)
                            if score > 0:
                                has_valid_audio = True
                
                # 计算平均值
                average_emotions = {
                    emotion: round(sum(scores) / len(scores), 2) 
                    for emotion, scores in all_emotions.items()
                }
                
                # 找到得分最高的情绪作为主要情绪标签
                if has_valid_audio and any(score > 0 for score in average_emotions.values()):
                    main_emotion = max(average_emotions.items(), key=lambda x: x[1])[0]
                else:
                    main_emotion = '无'  # 如果所有文件都是静音或情绪得分都为0，则标记为'无'
                
                emotion_results['总体统计'] = {
                    '情绪标签': main_emotion,
                    '各项情绪得分': average_emotions
                }
            else:
                # 如果是静音文件，情绪值设为0
                default_emotions = {
                    '中性': 0,
                    '快乐': 0,
                    '恐惧': 0,
                    '悲伤': 0,
                    '惊讶': 0,
                    '愤怒': 0,
                    '平静': 0,
                    '厌恶': 0
                }
                emotion_results['总体统计'] = {
                    '情绪标签': '暂无有效的情绪分析数据。',
                    '各项情绪得分': default_emotions
                }
            
            print("语音情绪分析完成")
            
        except Exception as e:
            print(f"执行语音情绪分析时出现异常: {e}")
            emotion_results = {}
    
    # 提取person_info，供后续使用
    person_info = questionnaire_data.get('personInfo', {})
    
    # 第二步：生成HTML报告
    html_path = None
    try:
        # 使用本地函数，无需导入
        
        # 构造用于HTML报告的数据结构
        report_data = {
            'user_id': record_id,
            'questionnaire_results': questionnaire_data,
            'emotion_results': emotion_results
        }
        
        # 转换问卷数据格式以匹配HTML报告生成器的期望格式
        user_info = {
            'GHQ12总分': questionnaire_data.get('一般健康问卷（GHQ-12）_一般健康问卷（GHQ-12）>总分', {}).get('score', 0),
            'PERMA总分': questionnaire_data.get('PERMA幸福量表_PERMA幸福量表>总分', {}).get('score', 0),
            '工种': person_info.get('工种', '普通职工')
        }
        
        html_path = generate_html_report_from_data(record_id, emotion_results, record_dir, False, user_info)
        
        if html_path and os.path.exists(html_path):
            print(f"HTML报告生成成功: {html_path}")
        else:
            print("HTML报告生成失败")
            
    except ImportError as e:
        print(f"错误：无法导入HTML报告生成模块: {e}")
    except Exception as e:
        print(f"生成HTML报告时出现异常: {e}")
    
    # 第三步：生成最终的JSON报告
    try:
        output_dir = kwargs.get('output_dir', record_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载心理评估报告模板
        template_data = load_template_data()
        
        # 获取心理风险评估和幸福感评估描述
        ghq12_score = questionnaire_data.get('一般健康问卷（GHQ-12）_一般健康问卷（GHQ-12）>总分', {}).get('score', 0)
        perma_score = questionnaire_data.get('PERMA幸福量表_PERMA幸福量表>总分', {}).get('score', 0)
        
        psychological_risk_assessment = get_ghq12_assessment(ghq12_score, template_data)
        happiness_assessment = get_perma_assessment(perma_score, template_data)
        
        # 获取多维度分析描述
        job_type = person_info.get('工种', '普通职工')
        psychology_results = emotion_results.get('三维度结果', {})
        multidimensional_analysis = get_dimension_analysis_text(psychology_results, job_type, template_data)
        
        # 构造最终报告数据
        final_report = {
            'personInfo': questionnaire_data.get('personInfo', {}),
            'emotion_analysis': {
                '整体情况': {
                    '心理风险评估-GHQ12': psychological_risk_assessment,
                    '幸福感评估-PERMA': happiness_assessment
                },
                '情绪谱图分析': emotion_results.get('融合情绪结果', {}),
                '多维度分析': multidimensional_analysis,
                '备注': '本测评基于科学的评估体系，用于反映个体心理状态的大致情况，不具备临床诊断效力。\n若您对测评结果持有疑虑，建议及时前往正规医疗机构进行专业诊疗。'
            },
            'html_report_path': os.path.abspath(html_path) if html_path and os.path.exists(html_path) else None
        }
        
        # 保存最终报告
        final_report_path = os.path.join(output_dir, f'{record_id}_complete_report.json')
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"完整报告已保存: {final_report_path}")
        return final_report_path
        
    except Exception as e:
        print(f"生成最终报告时出现异常: {e}")
        return None

def generate_complete_report(user_id, **kwargs):
    """
    为指定用户ID生成完整的心理评估报告（包含情绪分析和HTML报告）
    保持原有功能以兼容现有代码
    
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
        tuple: (HTML报告文件路径, PDF报告文件路径)，失败时返回(None, None)
    """
    print(f"开始为用户 '{user_id}' 生成完整的心理评估报告...")
    
    # 第一步：直接调用情绪分析函数，获取内存中的结果
    try:
        print("开始情绪分析...")
        
        # 准备参数
        data_dir = kwargs.get('data_dir', 'E:\\200_main_business\\朔黄工程\\第二阶段数据\\data')
        user_report_path = kwargs.get('user_report', 'E:\\200_main_business\\朔黄工程\\第二阶段数据\\data\\用户综合报告.json')
        model_path = kwargs.get('model_path')
        device = kwargs.get('device', 'auto')
        exclude_labels = kwargs.get('exclude_labels')
        include_labels = kwargs.get('include_labels')
        
        # 执行情绪分析，返回结果字典而不保存文件
        emotion_results = analyze_emotion_for_user(
            user_id=user_id,
            data_dir=data_dir,
            user_report=user_report_path,
            model_path=model_path,
            device=device,
            exclude_labels=exclude_labels,
            include_labels=include_labels
        )
        
        if emotion_results is None:
            print("情绪分析失败")
            return None, None
        
        print("情绪分析完成")
        
    except Exception as e:
        print(f"执行情绪分析时出现异常: {e}")
        return None, None
    
    # 第二步：生成HTML报告
    try:
        # 使用本地函数，无需导入
        # 提供默认的用户数据
        default_user_info = {
            'GHQ12总分': 0,
            'PERMA总分': 50,
            '工种': '普通职工'
        }
        html_path = generate_html_report_from_data(user_id, emotion_results, "results", False, default_user_info)
        
        if html_path and os.path.exists(html_path):
            print(f"HTML报告生成成功: {html_path}")
        else:
            print("HTML报告生成失败")
            return None, None
            
    except ImportError as e:
        print(f"错误：无法导入HTML报告生成模块: {e}")
        return None, None
    except Exception as e:
        print(f"生成HTML报告时出现异常: {e}")
        return None, None
    
    # 返回HTML报告路径
    try:
        return html_path
            
    except Exception as e:
        print(f"生成PDF报告时出现异常: {e}")
        print("PDF报告生成失败，但HTML报告已生成")
        return html_path, None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='为指定用户或recordId生成完整的心理评估报告（情绪分析 + 可视化图表 + HTML报告）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用集成模式（JSON文件）
  python generate_complete_report.py --json_file data.json
  python generate_complete_report.py --json_file data.json --cleanup_temp_files
  python generate_complete_report.py --json_file data.json --no_download_audio
  
  # 使用原有方式（用户ID）
  python generate_complete_report.py --user_id 飞雪无痕
  python generate_complete_report.py --user_id 飞雪无痕 --device cuda
  
  # 使用新方式（recordId从temp_data目录）
  python generate_complete_report.py --record_id 32525
  python generate_complete_report.py --record_id 32525 --device cuda
  python generate_complete_report.py --record_id 32525 --output_dir ./my_results
        """
    )
    
    # 互斥参数组：user_id、record_id 或 json_file
    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument('--user_id', type=str, 
                         help='用户ID（用于原有数据处理方式）')
    id_group.add_argument('--record_id', type=str, 
                         help='记录ID（用于从temp_data目录处理数据）')
    id_group.add_argument('--json_file', type=str, 
                         help='JSON数据文件路径（用于集成模式直接处理）')
    
    # 可选参数
    parser.add_argument('--data_dir', type=str, 
                       default='E:\\200_main_business\\朔黄工程\\第二阶段数据\\data',
                       help='数据目录路径（仅用于user_id模式）')
    
    parser.add_argument('--user_report', type=str, 
                       default='E:\\200_main_business\\朔黄工程\\第二阶段数据\\data\\用户综合报告.json',
                       help='用户综合报告JSON文件路径（仅用于user_id模式）')
    
    parser.add_argument('--temp_data_dir', type=str, 
                       default='E:\\200_main_business\\朔黄工程\\第二阶段数据\\code\\语音端到端demo-6\\temp_data',
                       help='temp_data目录路径（仅用于record_id模式）')
    
    parser.add_argument('--model_path', type=str, 
                       help='模型路径（默认使用内置路径）')
    
    parser.add_argument('--output_dir', type=str, 
                       help='输出目录（默认为results或record目录）')
    
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'],
                       help='计算设备: cpu 或 cuda（默认: cpu）')
    
    parser.add_argument('--exclude_labels', type=str, nargs='*', 
                       default=['预热'],
                       help='排除的标签列表（默认: 预热）')
    
    parser.add_argument('--include_labels', type=str, nargs='*', 
                       help='只包含的标签列表（如果指定，则只处理这些标签的音频）')
    
    # 集成模式专用参数
    parser.add_argument('--temp_base_dir', type=str, 
                       help='临时文件基础目录（仅用于json_file模式）')
    
    parser.add_argument('--no_download_audio', action='store_true', 
                       help='跳过音频下载，仅处理问卷数据（仅用于json_file模式）')
    
    parser.add_argument('--cleanup_temp_files', action='store_true', 
                       help='处理完成后清理临时音频文件（仅用于json_file模式）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("完整心理评估报告生成器")
    print("=" * 60)
    
    # 根据参数选择处理模式
    if args.json_file:
        # 集成模式：直接从JSON文件处理
        print(f"处理模式: 集成模式（JSON文件）")
        print(f"JSON文件: {args.json_file}")
        
        # 构建参数字典，过滤掉None值和不相关的参数
        kwargs = {k: v for k, v in vars(args).items() 
                  if k not in ['json_file', 'user_id', 'record_id', 'data_dir', 'user_report', 'temp_data_dir'] and v is not None}
        
        print(f"输出目录: {kwargs.get('output_dir', '默认(当前目录)')}") 
        print(f"临时目录: {kwargs.get('temp_base_dir', '默认(系统临时目录)')}")
        print(f"计算设备: {kwargs.get('device', 'cpu')}")
        print(f"音频下载: {'跳过' if kwargs.get('no_download_audio') else '启用'}")
        print(f"清理临时文件: {'是' if kwargs.get('cleanup_temp_files') else '否'}")
        print(f"可视化功能: {'启用' if MATPLOTLIB_AVAILABLE else '禁用(matplotlib未安装)'}")
        pdf_options = []
        if WEASYPRINT_AVAILABLE:
            pdf_options.append('WeasyPrint')
        if PDFKIT_AVAILABLE:
            pdf_options.append('pdfkit')
        pdf_status = '/'.join(pdf_options) if pdf_options else '禁用'
        print(f"PDF生成: {pdf_status}")
        print("="*60)
        
        # 生成完整报告
        report_path = generate_complete_report_from_json(args.json_file, **kwargs)
        
        print("=" * 60)
        if report_path:
            print(f"✅ 完整报告生成成功!")
            print(f"📄 JSON报告位置: {os.path.abspath(report_path)}")
            
            # 尝试读取并显示报告中的HTML路径
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                html_path = report_data.get('html_report_path')
                if html_path:
                    print(f"📄 HTML报告位置: {os.path.abspath(html_path)}")
            except:
                pass
        else:
            print("❌ 报告生成失败")
            print("请检查错误信息并重试")
        print("=" * 60)
        
        return 0 if report_path else 1
        
    elif args.record_id:
        # 新模式：从temp_data目录处理recordId
        print(f"处理模式: temp_data目录")
        print(f"记录ID: {args.record_id}")
        print(f"temp_data目录: {args.temp_data_dir}")
        
        # 构建参数字典，过滤掉None值和不相关的参数
        kwargs = {k: v for k, v in vars(args).items() 
                  if k not in ['record_id', 'user_id', 'data_dir', 'user_report'] and v is not None}
        
        print(f"输出目录: {kwargs.get('output_dir', '默认(record目录)')}")
        print(f"计算设备: {kwargs.get('device', 'cpu')}")
        print(f"可视化功能: {'启用' if MATPLOTLIB_AVAILABLE else '禁用(matplotlib未安装)'}")
        pdf_options = []
        if WEASYPRINT_AVAILABLE:
            pdf_options.append('WeasyPrint')
        if PDFKIT_AVAILABLE:
            pdf_options.append('pdfkit')
        pdf_status = '/'.join(pdf_options) if pdf_options else '禁用'
        print(f"PDF生成: {pdf_status}")
        print("="*60)
        
        # 生成完整报告
        report_path = generate_complete_report_from_temp_data(args.record_id, **kwargs)
        
        print("=" * 60)
        if report_path:
            print(f"✅ 完整报告生成成功!")
            print(f"📄 JSON报告位置: {os.path.abspath(report_path)}")
            
            # 尝试读取并显示报告中的HTML路径
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                html_path = report_data.get('html_report_path')
                if html_path:
                    print(f"📄 HTML报告位置: {os.path.abspath(html_path)}")
            except:
                pass
        else:
            print("❌ 报告生成失败")
            print("请检查错误信息并重试")
        print("=" * 60)
        
        return 0 if report_path else 1
        
    else:
        # 原有模式：使用user_id
        print(f"处理模式: 原有数据处理")
        print(f"用户ID: {args.user_id}")
        
        # 构建参数字典，过滤掉None值和不相关的参数
        kwargs = {k: v for k, v in vars(args).items() 
                  if k not in ['user_id', 'record_id', 'temp_data_dir'] and v is not None}
        
        print(f"数据目录: {kwargs.get('data_dir', '默认')}")
        print(f"输出目录: {kwargs.get('output_dir', '默认(results)')}")
        print(f"计算设备: {kwargs.get('device', 'cpu')}")
        print(f"可视化功能: {'启用' if MATPLOTLIB_AVAILABLE else '禁用(matplotlib未安装)'}")
        pdf_options = []
        if WEASYPRINT_AVAILABLE:
            pdf_options.append('WeasyPrint')
        if PDFKIT_AVAILABLE:
            pdf_options.append('pdfkit')
        pdf_status = '/'.join(pdf_options) if pdf_options else '禁用'
        print(f"PDF生成: {pdf_status}")
        print("="*60)
        
        # 生成完整报告
        html_path = generate_complete_report(args.user_id, **kwargs)
        
        print("=" * 60)
        if html_path:
            print(f"✅ 完整报告生成成功!")
            print(f"📄 HTML报告位置: {os.path.abspath(html_path)}")
        else:
            print("❌ 报告生成失败")
            print("请检查错误信息并重试")
        print("=" * 60)
        
        return 0 if html_path else 1

if __name__ == "__main__":
    sys.exit(main())