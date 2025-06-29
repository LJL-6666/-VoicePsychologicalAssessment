<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-06-15 23:15:20
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-06-16 09:30:48
 * @FilePath: \朔黄工程\第二阶段数据\code\语音端到端demo-6\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 语音端到端情绪系统

本项目实现了一个端到端的语音情绪识别，包括数据处理和情绪预测以及报告输出等功能。
输入一个人的请求ID，会索引到该用户的音频文件夹下的所有音频【排除预热标签的音频】，输出一个人的8个情绪得分和3维度量表得分，并生成HTML心理评估报告和json文件。

python generate_complete_report.py --json_file "E:\\200_main_business\\朔黄工程\\第二阶段数据\\data\\测评包数据结构_1.txt" --cleanup_temp_files

python generate_complete_report.py --json_file "E:\\200_main_business\\朔黄工程\\第二阶段数据\\data\\测评包数据结构_2.txt" --cleanup_temp_files
-----------------------------
python start_concurrent_processor.py

-----------------------------


## PDF生成依赖安装

为了生成PDF报告，需要安装以下依赖之一：

### 方法1：使用weasyprint（推荐）
```bash
pip install weasyprint
```

### 方法2：使用pdfkit
```bash
pip install pdfkit
```

**注意：** 使用pdfkit还需要安装wkhtmltopdf工具：
- Windows: 下载并安装 https://wkhtmltopdf.org/downloads.html
- Linux: `sudo apt-get install wkhtmltopdf`
- macOS: `brew install wkhtmltopdf`

如果不安装PDF依赖，程序仍会正常运行并生成HTML报告，只是不会生成PDF文件。


## 环境依赖
Anaconda 3
Python 3.8
Pytorch 1.13.1


```bash
# 创建Anaconda环境
conda create -n ml python=3.10

# 激活环境
conda activate ml

# 基本依赖
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

python -m pip install mser -U -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
pip install -r requirements.txt
```
## 注意事项
1. 本项目使用相对路径，确保不要修改目录结构
2. 预测脚本会自动处理路径问题，即使config.py导入失败也能正常工作
3. 如果使用Emotion2Vec特征提取方法，请确保安装了funasr库

