#!/usr/bin/env python
"""
简单的训练示例脚本
演示如何训练Audio-GS模型
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_audio():
    """创建一个测试音频文件用于演示"""
    import torchaudio

    print("创建测试音频...")

    # 创建samples目录
    os.makedirs("samples", exist_ok=True)

    # 生成测试音频：组合正弦波
    sample_rate = 44100
    duration = 2.0  # 2秒
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # 创建和弦：C大调 (C-E-G)
    frequencies = [261.63, 329.63, 392.00]  # Hz
    audio = torch.zeros_like(t)

    for freq in frequencies:
        audio += 0.3 * torch.sin(2 * np.pi * freq * t)

    # 添加音量包络
    envelope = torch.exp(-t / 0.5)  # 指数衰减
    audio = audio * envelope

    # 归一化
    audio = audio / torch.max(torch.abs(audio))

    # 保存
    output_path = "samples/test_chord.wav"
    torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)

    print(f"✓ 测试音频已保存到: {output_path}")
    print(f"  时长: {duration}秒")
    print(f"  采样率: {sample_rate}Hz")

    return output_path

def train_simple():
    """简单训练示例"""
    print("\n" + "="*50)
    print("Audio-GS 简单训练示例")
    print("="*50)

    # 检查是否有测试音频
    test_audio = "samples/test_chord.wav"
    if not os.path.exists(test_audio):
        test_audio = create_test_audio()

    # 训练参数
    config = {
        "input_path": test_audio,
        "num_gaussians": 100,      # 使用较少的高斯以加快训练
        "num_steps": 1000,          # 较少的步数用于快速演示
        "lr": 0.01,
        "log_dir": "logs/test_example",
        "save_steps": 500,
    }

    print("\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 构建命令
    cmd = f"python main.py"
    for key, value in config.items():
        cmd += f" --{key} {value}"

    print(f"\n执行命令:\n{cmd}")

    # 执行训练
    print("\n开始训练...")
    os.system(cmd)

    print("\n训练完成！")
    print(f"结果保存在: {config['log_dir']}/")

def train_with_different_configs():
    """使用不同配置训练同一音频"""
    print("\n" + "="*50)
    print("多配置对比训练")
    print("="*50)

    # 确保有测试音频
    test_audio = "samples/test_chord.wav"
    if not os.path.exists(test_audio):
        test_audio = create_test_audio()

    # 不同的配置
    configs = [
        {
            "name": "low_quality",
            "num_gaussians": 50,
            "num_steps": 500,
            "description": "低质量快速版"
        },
        {
            "name": "medium_quality",
            "num_gaussians": 200,
            "num_steps": 2000,
            "description": "中等质量"
        },
        {
            "name": "high_quality",
            "num_gaussians": 500,
            "num_steps": 5000,
            "description": "高质量"
        }
    ]

    for config in configs:
        print(f"\n训练配置: {config['description']}")
        print(f"  高斯数: {config['num_gaussians']}")
        print(f"  训练步数: {config['num_steps']}")

        cmd = f"python main.py --input_path {test_audio}"
        cmd += f" --num_gaussians {config['num_gaussians']}"
        cmd += f" --num_steps {config['num_steps']}"
        cmd += f" --log_dir logs/comparison_{config['name']}"

        print(f"执行: {cmd}")
        os.system(cmd)

    print("\n所有训练完成！")
    print("查看结果对比:")
    print("  logs/comparison_low_quality/")
    print("  logs/comparison_medium_quality/")
    print("  logs/comparison_high_quality/")

def train_music_example():
    """音乐压缩训练示例"""
    print("\n" + "="*50)
    print("音乐压缩训练示例")
    print("="*50)

    # 创建更复杂的音乐样本
    print("创建音乐样本...")

    sample_rate = 44100
    duration = 3.0
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # 创建旋律
    melody_times = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    melody_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00]  # C-D-E-F-G-A

    audio = torch.zeros_like(t)

    for i, (start_time, freq) in enumerate(zip(melody_times, melody_freqs)):
        # 创建音符
        note_mask = (t >= start_time) & (t < start_time + 0.4)
        note_t = t[note_mask] - start_time

        # 基频 + 谐波
        note = torch.zeros_like(note_t)
        for harmonic in range(1, 4):
            amplitude = 0.5 / harmonic
            note += amplitude * torch.sin(2 * np.pi * freq * harmonic * note_t)

        # ADSR包络
        attack = torch.minimum(note_t / 0.05, torch.ones_like(note_t))
        decay = torch.exp(-3 * (note_t - 0.05))
        envelope = attack * decay

        audio[note_mask] += note * envelope

    # 添加和弦背景
    chord_freqs = [130.81, 164.81, 196.00]  # C3-E3-G3
    for freq in chord_freqs:
        audio += 0.1 * torch.sin(2 * np.pi * freq * t) * torch.exp(-t / 2)

    # 归一化
    audio = audio / torch.max(torch.abs(audio))

    # 保存
    music_path = "samples/test_music.wav"
    os.makedirs("samples", exist_ok=True)
    torchaudio.save(music_path, audio.unsqueeze(0), sample_rate)

    print(f"✓ 音乐样本已创建: {music_path}")

    # 使用音乐配置训练
    cmd = f"python main.py --input_path {music_path}"
    cmd += " --config configs/music.yaml"
    cmd += " --log_dir logs/music_example"

    print(f"\n使用音乐优化配置训练...")
    print(f"命令: {cmd}")
    os.system(cmd)

    print("\n音乐训练完成！")

def analyze_results():
    """分析训练结果"""
    print("\n" + "="*50)
    print("分析训练结果")
    print("="*50)

    log_dirs = list(Path("logs").glob("*/"))

    if not log_dirs:
        print("没有找到训练结果。请先运行训练。")
        return

    print(f"找到 {len(log_dirs)} 个训练结果:\n")

    for log_dir in log_dirs:
        log_file = log_dir / "log_train.txt"
        if not log_file.exists():
            continue

        print(f"目录: {log_dir.name}")

        # 读取最后几行日志获取SNR
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-20:]):
                if "SNR:" in line or "Compression" in line or "Bitrate:" in line:
                    print(f"  {line.strip()}")

        # 检查输出文件
        wav_file = log_dir / "train" / "reconstructed.wav"
        if wav_file.exists():
            size_kb = wav_file.stat().st_size / 1024
            print(f"  重建音频: {size_kb:.2f} KB")

        print()

def main():
    print("Audio-GS 训练示例脚本")
    print("="*50)
    print("\n选择训练模式:")
    print("1. 简单训练示例")
    print("2. 多配置对比训练")
    print("3. 音乐压缩示例")
    print("4. 分析已有结果")
    print("5. 运行所有示例")

    choice = input("\n请选择 (1-5): ").strip()

    if choice == "1":
        train_simple()
    elif choice == "2":
        train_with_different_configs()
    elif choice == "3":
        train_music_example()
    elif choice == "4":
        analyze_results()
    elif choice == "5":
        train_simple()
        train_with_different_configs()
        train_music_example()
        analyze_results()
    else:
        print("无效选择，运行默认简单训练...")
        train_simple()

    print("\n完成！查看 logs/ 目录获取结果。")

if __name__ == "__main__":
    # 检查是否在正确目录
    if not os.path.exists("main.py"):
        print("错误：请在 Audio-GS 项目根目录运行此脚本")
        print("cd F:/Code/Audio-GS")
        sys.exit(1)

    # 检查依赖
    try:
        import torchaudio
    except ImportError:
        print("错误：请先安装依赖")
        print("pip install -r requirements.txt")
        sys.exit(1)

    main()