# Audio-GS: Content-Adaptive Audio Representation via 2D Gaussians

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English](#english) | [中文](#中文)

</div>

## English

A novel audio compression method using 2D Gaussian representation in the time-frequency domain, inspired by [Image-GS](https://github.com/NYU-ICL/image-gs).

### 🌟 Key Features

- **🎯 Content-Adaptive**: Automatically allocates Gaussians based on spectral importance
- **⚡ Ultra-Fast Decoding**: ~0.3K MACs per sample, suitable for embedded devices
- **📊 Flexible Quality**: Adjustable compression ratio from 10x to 100x
- **🔧 Easy to Use**: Simple CLI interface with pre-configured settings

### 🚀 Quick Start

#### One-Click Setup and Demo
```bash
# Automatic setup, test, and demo training
python quick_start.py
```

#### Manual Installation
```bash
# Create environment
conda create -n audio-gs python=3.10
conda activate audio-gs

# Install dependencies
pip install -r requirements.txt

# Run test
python test_simple.py
```

### 📖 Basic Usage

#### Simple Training
```bash
# Train with default settings
python main.py --input_path your_audio.wav --num_gaussians 500

# Train with specific config
python main.py --input_path music.wav --config configs/music.yaml

# Train with quantization
python main.py --input_path audio.wav --quantize --bits_per_param 8
```

#### Windows Users
```bash
# Interactive training script
run_training.bat
```

### 🎛️ Configuration

| Audio Type | Gaussians | Config File | Typical Bitrate |
|------------|-----------|-------------|-----------------|
| Speech | 200-300 | `configs/speech.yaml` | 8-16 kbps |
| Music | 500-1000 | `configs/music.yaml` | 32-64 kbps |
| Complex | 1000-2000 | Custom | 64-128 kbps |

### 📊 Performance Benchmarks

| Method | Compression Ratio | SNR (dB) | Decode Speed |
|--------|-------------------|----------|--------------|
| Audio-GS (Speech) | 100x | 20-25 | Real-time |
| Audio-GS (Music) | 50x | 22-28 | Real-time |
| MP3 128k | 11x | 35+ | Real-time |
| Opus 32k | 44x | 30+ | Real-time |

### 🔬 Technical Details

Audio-GS represents audio signals as a weighted sum of 2D Gaussians in the time-frequency domain:

```
S(t,f) = Σᵢ αᵢ · exp(-½[(t-μₜᵢ)²/σₜᵢ² + (f-μfᵢ)²/σfᵢ²])
```

Where each Gaussian is parameterized by:
- **Position**: (μₜ, μf) - time and frequency centers
- **Spread**: (σₜ, σf) - temporal and spectral widths
- **Weight**: α - contribution magnitude
- **Phase**: φ - phase information

### 📁 Project Structure

```
Audio-GS/
├── main.py              # Main training/inference script
├── model.py             # Core AudioGaussian2D model
├── quick_start.py       # One-click setup and demo
├── train_example.py     # Training examples
├── test_simple.py       # Unit tests
├── TRAINING_GUIDE.md    # Detailed training guide
├── configs/            # Configuration presets
│   ├── default.yaml
│   ├── music.yaml
│   └── speech.yaml
├── utils/              # Utility functions
│   ├── audio_utils.py
│   ├── gaussian_utils.py
│   ├── quantization_utils.py
│   └── misc_utils.py
├── samples/            # Audio samples (create this)
└── logs/               # Training outputs (auto-created)
```

---

## 中文

基于[Image-GS](https://github.com/NYU-ICL/image-gs)思想的新型音频压缩方法，使用时频域2D高斯表示。

### 🌟 核心特性

- **🎯 内容自适应**: 根据频谱重要性自动分配高斯
- **⚡ 超快解码**: 每采样点仅需~0.3K MACs，适合嵌入式设备
- **📊 灵活质量**: 压缩率可调节（10x到100x）
- **🔧 简单易用**: 简洁的命令行界面和预配置

### 🚀 快速开始

#### 一键安装和演示
```bash
# 自动安装、测试和演示训练
python quick_start.py
```

#### 手动安装
```bash
# 创建环境
conda create -n audio-gs python=3.10
conda activate audio-gs

# 安装依赖
pip install -r requirements.txt

# 运行测试
python test_simple.py
```

### 📖 基础用法

#### 简单训练
```bash
# 使用默认设置训练
python main.py --input_path your_audio.wav --num_gaussians 500

# 使用特定配置训练
python main.py --input_path music.wav --config configs/music.yaml

# 带量化的训练
python main.py --input_path audio.wav --quantize --bits_per_param 8
```

#### Windows用户
```bash
# 交互式训练脚本
run_training.bat
```

### 🎛️ 配置说明

| 音频类型 | 高斯数量 | 配置文件 | 典型比特率 |
|---------|---------|----------|-----------|
| 语音 | 200-300 | `configs/speech.yaml` | 8-16 kbps |
| 音乐 | 500-1000 | `configs/music.yaml` | 32-64 kbps |
| 复杂音频 | 1000-2000 | 自定义 | 64-128 kbps |

### 📊 性能基准

| 方法 | 压缩率 | 信噪比 (dB) | 解码速度 |
|------|--------|-------------|----------|
| Audio-GS (语音) | 100x | 20-25 | 实时 |
| Audio-GS (音乐) | 50x | 22-28 | 实时 |
| MP3 128k | 11x | 35+ | 实时 |
| Opus 32k | 44x | 30+ | 实时 |

### 🔬 技术原理

Audio-GS将音频信号表示为时频域中2D高斯的加权和：

```
S(t,f) = Σᵢ αᵢ · exp(-½[(t-μₜᵢ)²/σₜᵢ² + (f-μfᵢ)²/σfᵢ²])
```

每个高斯由以下参数定义：
- **位置**: (μₜ, μf) - 时间和频率中心
- **扩展**: (σₜ, σf) - 时间和频率宽度
- **权重**: α - 贡献幅度
- **相位**: φ - 相位信息

### 🎯 训练指南

详细的训练说明请查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

#### 快速训练示例

```bash
# 语音压缩（高压缩率）
python main.py \
    --input_path speech.wav \
    --config configs/speech.yaml \
    --num_gaussians 300 \
    --quantize

# 音乐压缩（高质量）
python main.py \
    --input_path music.wav \
    --config configs/music.yaml \
    --num_gaussians 1000 \
    --num_steps 10000

# 自适应高斯添加
python main.py \
    --input_path complex.wav \
    --num_gaussians 500 \
    --adaptive_add \
    --add_gaussians_steps 1000 2000 3000
```

### 🔍 评估和可视化

```bash
# 从检查点重建音频
python main.py \
    --eval \
    --ckpt_file logs/your_model/checkpoints/checkpoint_005000.pth \
    --log_dir eval_output

# 查看训练日志
tail -f logs/your_model/log_train.txt
```

### 📈 性能优化建议

1. **提高质量**: 增加高斯数量和训练步数
2. **减小文件**: 启用量化（`--quantize`）和减少高斯数
3. **加快训练**: 使用GPU并调整学习率
4. **特定优化**: 根据音频类型选择合适的配置文件

### 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 📄 许可证

MIT License - 可自由用于学术和商业项目

### 🙏 致谢

- [Image-GS](https://github.com/NYU-ICL/image-gs) - 提供了高斯表示的灵感
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [librosa](https://librosa.org/) - 音频处理库
- [torchaudio](https://pytorch.org/audio/) - PyTorch音频扩展

### 📮 联系

- GitHub Issues: [提交问题](https://github.com/yourusername/Audio-GS/issues)
- 技术讨论: [Discussions](https://github.com/yourusername/Audio-GS/discussions)

---

<div align="center">

**如果这个项目对你有帮助，请给个⭐星标支持！**

Made with ❤️ for audio compression research

</div>