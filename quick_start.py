#!/usr/bin/env python
"""
Audio-GS 快速开始脚本
一键安装依赖、创建测试音频并开始训练
"""

import os
import sys
import subprocess

def check_python_version():
    """检查Python版本"""
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    print(f"✓ Python版本: {sys.version}")
    return True

def install_dependencies():
    """安装依赖包"""
    print("\n安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败")
        print("请手动运行: pip install -r requirements.txt")
        return False

def create_demo_audio():
    """创建演示音频"""
    print("\n创建演示音频...")
    try:
        import torch
        import torchaudio
        import numpy as np

        os.makedirs("samples", exist_ok=True)

        # 1. 简单和弦
        print("  创建和弦音频...")
        sr = 44100
        duration = 2.0
        t = torch.linspace(0, duration, int(sr * duration))

        # C大调和弦
        freqs = [261.63, 329.63, 392.00]
        chord = sum(0.3 * torch.sin(2 * np.pi * f * t) for f in freqs)
        chord *= torch.exp(-t / 0.5)  # 衰减
        chord = chord / torch.max(torch.abs(chord))

        torchaudio.save("samples/chord.wav", chord.unsqueeze(0), sr)

        # 2. 语音模拟（调制正弦波）
        print("  创建语音模拟...")
        carrier = 200  # 载波频率
        modulator = 5   # 调制频率
        speech = torch.sin(2 * np.pi * carrier * t) * (1 + 0.5 * torch.sin(2 * np.pi * modulator * t))
        speech *= torch.exp(-((t - 1) ** 2) / 0.5)  # 高斯包络
        speech = speech / torch.max(torch.abs(speech))

        torchaudio.save("samples/speech_sim.wav", speech.unsqueeze(0), sr)

        # 3. 音乐片段
        print("  创建音乐片段...")
        notes = []
        note_times = np.linspace(0, 2, 8)
        note_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 349.23, 329.63, 293.66]

        music = torch.zeros_like(t)
        for start, freq in zip(note_times, note_freqs):
            mask = (t >= start) & (t < start + 0.3)
            note_t = t[mask] - start
            note = torch.sin(2 * np.pi * freq * note_t)
            note *= torch.exp(-5 * note_t)  # 快速衰减
            music[mask] += note

        music = music / torch.max(torch.abs(music))
        torchaudio.save("samples/melody.wav", music.unsqueeze(0), sr)

        print("✓ 创建了3个演示音频:")
        print("  - samples/chord.wav (和弦)")
        print("  - samples/speech_sim.wav (语音模拟)")
        print("  - samples/melody.wav (旋律)")
        return True

    except Exception as e:
        print(f"❌ 创建音频失败: {e}")
        return False

def run_test():
    """运行测试"""
    print("\n运行测试...")
    try:
        result = subprocess.run([sys.executable, "test_simple.py"], capture_output=True, text=True)
        if "All tests passed" in result.stdout:
            print("✓ 测试通过")
            return True
        else:
            print("❌ 测试失败")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return False

def quick_train():
    """快速训练示例"""
    print("\n开始快速训练示例...")

    configs = [
        ("samples/chord.wav", 100, 1000, "快速测试"),
        ("samples/speech_sim.wav", 200, 2000, "语音压缩"),
        ("samples/melody.wav", 300, 3000, "音乐压缩"),
    ]

    for audio_path, n_gaussians, n_steps, desc in configs:
        if not os.path.exists(audio_path):
            continue

        print(f"\n训练 {desc}:")
        print(f"  文件: {audio_path}")
        print(f"  高斯数: {n_gaussians}")
        print(f"  步数: {n_steps}")

        cmd = [
            sys.executable, "main.py",
            "--input_path", audio_path,
            "--num_gaussians", str(n_gaussians),
            "--num_steps", str(n_steps),
            "--log_dir", f"logs/quick_{os.path.basename(audio_path).split('.')[0]}"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {desc} 完成")
        except subprocess.CalledProcessError:
            print(f"❌ {desc} 失败")

def show_results():
    """显示结果"""
    print("\n" + "="*50)
    print("训练结果:")
    print("="*50)

    from pathlib import Path
    log_dirs = list(Path("logs").glob("quick_*/"))

    if not log_dirs:
        print("暂无结果")
        return

    for log_dir in log_dirs:
        print(f"\n{log_dir.name}:")

        # 查找重建的音频
        wav_files = list(log_dir.glob("**/*.wav"))
        if wav_files:
            for wav in wav_files:
                size_kb = wav.stat().st_size / 1024
                print(f"  输出: {wav.name} ({size_kb:.1f} KB)")

        # 读取日志中的SNR
        log_file = log_dir / "log_train.txt"
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines[-10:]):
                    if "SNR:" in line:
                        print(f"  {line.strip()}")
                        break

def main():
    print("="*50)
    print("Audio-GS 快速开始")
    print("="*50)

    # 1. 检查环境
    print("\n[1/5] 检查环境...")
    if not check_python_version():
        return 1

    # 2. 安装依赖
    print("\n[2/5] 安装依赖...")
    if not os.path.exists("requirements.txt"):
        print("❌ 找不到requirements.txt")
        print("请确保在Audio-GS项目目录下运行")
        return 1

    try:
        import torch
        import torchaudio
        print("✓ 依赖已安装")
    except ImportError:
        if not install_dependencies():
            return 1

    # 3. 创建测试音频
    print("\n[3/5] 准备音频...")
    if not os.path.exists("samples"):
        os.makedirs("samples")

    existing_samples = list(Path("samples").glob("*.wav"))
    if not existing_samples:
        if not create_demo_audio():
            return 1
    else:
        print(f"✓ 找到 {len(existing_samples)} 个音频文件")

    # 4. 运行测试
    print("\n[4/5] 测试系统...")
    run_test()  # 即使失败也继续

    # 5. 快速训练
    print("\n[5/5] 运行训练...")
    quick_train()

    # 显示结果
    show_results()

    print("\n" + "="*50)
    print("✨ Audio-GS 已准备就绪！")
    print("="*50)
    print("\n下一步:")
    print("1. 将你的音频文件放入 samples/ 目录")
    print("2. 运行: python main.py --input_path samples/your_audio.wav --num_gaussians 500")
    print("3. 查看 logs/ 目录中的结果")
    print("\n查看 TRAINING_GUIDE.md 了解更多训练技巧")

    return 0

if __name__ == "__main__":
    # 检查是否在正确目录
    if not os.path.exists("main.py"):
        print("❌ 请在Audio-GS项目根目录运行此脚本")
        print("cd F:/Code/Audio-GS")
        sys.exit(1)

    sys.exit(main())