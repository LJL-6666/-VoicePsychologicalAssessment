#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行心理评估API演示
自动启动服务器并运行测试
"""

import subprocess
import time
import sys
import os
import signal

def run_demo():
    """运行完整演示"""
    print("=" * 60)
    print("心理评估API一键演示")
    print("=" * 60)
    
    # 启动API服务器
    print("\n1. 启动API服务器...")
    server_process = subprocess.Popen(
        [sys.executable, 'api_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    
    # 等待服务器启动
    print("   等待服务器启动...")
    time.sleep(3)
    
    # 检查服务器是否启动成功
    if server_process.poll() is not None:
        print("   ✗ 服务器启动失败")
        stdout, stderr = server_process.communicate()
        print(f"   错误输出: {stderr}")
        return
    
    print("   ✓ 服务器已启动")
    
    try:
        # 运行测试客户端
        print("\n2. 运行测试客户端...")
        client_process = subprocess.Popen(
            [sys.executable, 'test_api.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        
        # 实时显示客户端输出
        for line in client_process.stdout:
            print(f"   {line}", end='')
        
        client_process.wait()
        
        if client_process.returncode == 0:
            print("\n✓ 测试完成")
        else:
            print("\n✗ 测试失败")
            
    except KeyboardInterrupt:
        print("\n\n中断测试")
    finally:
        # 关闭服务器
        print("\n3. 关闭服务器...")
        if sys.platform == 'win32':
            # Windows
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(server_process.pid)], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Unix/Linux/Mac
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        
        server_process.wait()
        print("   ✓ 服务器已关闭")

if __name__ == '__main__':
    try:
        run_demo()
    except Exception as e:
        print(f"\n演示失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n演示结束")