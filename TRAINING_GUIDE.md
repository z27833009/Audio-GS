# Audio-GS 训练指南

## 快速开始

### 1. 安装环境

```bash
# 创建虚拟环境
conda create -n audio-gs python=3.10
conda activate audio-gs

# 安装依赖
cd F:/Code/Audio-GS
pip install -r requirements.txt
```

### 2. 准备音频文件

将你的音频文件放在 `samples/` 目录下，支持格式：
- WAV (推荐)
- MP3
- FLAC
- M4A

```bash
# 创建样本目录
mkdir samples
# 将音频文件复制到samples目录
```

### 3. 基础训练

```bash
# 最简单的训练命令
python main.py --input_path samples/your_audio.wav --num_gaussians 500

# 指定输出目录
python main.py --input_path samples/piano.wav --num_gaussians 500 --log_dir outputs/piano_test
```

## 详细训练参数

### 核心参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--num_gaussians` | 500 | 高斯数量，越多质量越好但文件越大 |
| `--num_steps` | 5000 | 训练步数，越多拟合越好 |
| `--lr` | 0.01 | 学习率，影响收敛速度 |
| `--init_method` | peaks | 初始化方法：peaks/random/uniform |

### 音频类型优化

#### 音乐训练
```bash
python main.py \
    --input_path samples/music.wav \
    --config configs/music.yaml \
    --num_gaussians 1000 \
    --num_steps 10000
```

#### 语音训练
```bash
python main.py \
    --input_path samples/speech.wav \
    --config configs/speech.yaml \
    --num_gaussians 300 \
    --num_steps 3000
```

#### 复杂音频（混音、交响乐）
```bash
python main.py \
    --input_path samples/symphony.wav \
    --num_gaussians 2000 \
    --num_steps 15000 \
    --adaptive_add \
    --add_gaussians_steps 2000 4000 6000 8000 \
    --lr 0.005
```

## 高级训练技巧

### 1. 自适应高斯添加
在训练过程中动态添加高斯到高误差区域：

```bash
python main.py \
    --input_path samples/complex_audio.wav \
    --num_gaussians 500 \
    --adaptive_add \
    --add_gaussians_steps 1000 2000 3000 \
    --num_steps 5000
```

### 2. 损失函数选择

- **L2损失**：适合一般音频
  ```bash
  --loss_type l2
  ```

- **频谱损失**：适合音乐，保留谐波结构
  ```bash
  --loss_type spectral --perceptual_weight 0.3
  ```

- **L1损失**：适合语音，保留清晰度
  ```bash
  --loss_type l1 --perceptual_weight 0.5
  ```

### 3. 量化压缩

训练后启用量化以减小文件大小：

```bash
# 8-bit量化（压缩率高）
python main.py \
    --input_path samples/audio.wav \
    --num_gaussians 500 \
    --quantize \
    --bits_per_param 8

# 16-bit量化（质量更好）
python main.py \
    --input_path samples/audio.wav \
    --num_gaussians 500 \
    --quantize \
    --bits_per_param 16
```

## 批量训练脚本

创建 `batch_train.py`：

```python
import os
import subprocess
from pathlib import Path

# 配置
SAMPLE_DIR = "samples"
OUTPUT_DIR = "outputs"
CONFIG_MAP = {
    "speech": "configs/speech.yaml",
    "music": "configs/music.yaml",
    "default": "configs/default.yaml"
}

def detect_audio_type(filename):
    """根据文件名猜测音频类型"""
    name_lower = filename.lower()
    if any(word in name_lower for word in ["speech", "voice", "talk"]):
        return "speech"
    elif any(word in name_lower for word in ["music", "song", "piano", "guitar"]):
        return "music"
    return "default"

def train_audio(audio_path, audio_type="default"):
    """训练单个音频文件"""
    audio_name = Path(audio_path).stem
    output_dir = os.path.join(OUTPUT_DIR, audio_name)

    cmd = [
        "python", "main.py",
        "--input_path", audio_path,
        "--config", CONFIG_MAP[audio_type],
        "--log_dir", output_dir
    ]

    print(f"Training {audio_name} with {audio_type} config...")
    subprocess.run(cmd)

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有音频文件
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(Path(SAMPLE_DIR).glob(ext))

    print(f"Found {len(audio_files)} audio files")

    # 训练每个文件
    for audio_file in audio_files:
        audio_type = detect_audio_type(audio_file.name)
        train_audio(str(audio_file), audio_type)

    print("Batch training completed!")

if __name__ == "__main__":
    main()
```

## 监控训练进度

### 查看日志
训练日志保存在 `log_dir` 目录下：
```
logs/your_audio_g500_peaks/
├── log_train.txt          # 训练日志
├── cfg_train.yaml         # 训练配置
├── train/                 # 训练输出
│   ├── reconstructed.wav  # 重建音频
│   └── gaussians_visualization.png  # 可视化
└── checkpoints/           # 模型检查点
    ├── checkpoint_001000.pth
    ├── checkpoint_002000.pth
    └── ...
```

### 实时监控
```bash
# 查看训练日志
tail -f logs/your_audio_g500_peaks/log_train.txt

# 使用tensorboard（如果添加了tensorboard支持）
tensorboard --logdir logs/
```

## 评估模型

### 从检查点重建音频
```bash
python main.py \
    --eval \
    --ckpt_file logs/your_audio_g500_peaks/checkpoints/checkpoint_005000.pth \
    --log_dir eval_output
```

### 计算压缩率
```python
# 查看压缩统计
Original size: 1720.32 KB
Compressed size: 23.44 KB
Compression ratio: 73.38x
Bitrate: 12.3 kbps
SNR: 24.56 dB
```

## 常见问题

### 1. 如何选择高斯数量？
- **语音**: 200-500个高斯
- **简单音乐**: 500-1000个高斯
- **复杂音乐**: 1000-3000个高斯
- **高保真**: 3000+个高斯

### 2. 训练不收敛？
- 降低学习率：`--lr 0.001`
- 增加训练步数：`--num_steps 10000`
- 尝试不同初始化：`--init_method uniform`

### 3. 重建质量差？
- 增加高斯数量
- 启用自适应添加：`--adaptive_add`
- 调整感知权重：`--perceptual_weight 0.5`

### 4. 文件太大？
- 启用量化：`--quantize --bits_per_param 6`
- 减少高斯数量
- 使用更激进的量化

## 性能基准

| 音频类型 | 高斯数 | 压缩率 | SNR | 比特率 |
|---------|--------|--------|-----|--------|
| 语音 | 300 | 100x | 20dB | 8kbps |
| 钢琴 | 500 | 60x | 25dB | 16kbps |
| 流行音乐 | 1000 | 30x | 22dB | 32kbps |
| 交响乐 | 2000 | 15x | 20dB | 64kbps |

## 下一步

1. **优化质量**: 调整损失函数权重和高斯数量
2. **减小文件**: 启用量化和熵编码
3. **批量处理**: 使用批处理脚本处理多个文件
4. **实时编码**: 开发流式编码器