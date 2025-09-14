# Audio-GS Training Guide

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
conda create -n audio-gs python=3.10
conda activate audio-gs

# Install dependencies
cd F:/Code/Audio-GS
pip install -r requirements.txt
```

### 2. Prepare Audio Files

Place your audio files in the `samples/` directory. Supported formats:
- WAV (recommended)
- MP3
- FLAC
- M4A

```bash
# Create samples directory
mkdir samples
# Copy your audio files to samples directory
```

### 3. Basic Training

```bash
# Simplest training command
python main.py --input_path samples/your_audio.wav --num_gaussians 500

# Specify output directory
python main.py --input_path samples/piano.wav --num_gaussians 500 --log_dir outputs/piano_test
```

## Training Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_gaussians` | 500 | Number of Gaussians (more = better quality, larger file) |
| `--num_steps` | 5000 | Training iterations (more = better fit) |
| `--lr` | 0.01 | Learning rate (affects convergence speed) |
| `--init_method` | peaks | Initialization method: peaks/random/uniform |

### Audio Type Optimization

#### Music Training
```bash
python main.py \
    --input_path samples/music.wav \
    --config configs/music.yaml \
    --num_gaussians 1000 \
    --num_steps 10000
```

#### Speech Training
```bash
python main.py \
    --input_path samples/speech.wav \
    --config configs/speech.yaml \
    --num_gaussians 300 \
    --num_steps 3000
```

#### Complex Audio (Orchestral, Mixed)
```bash
python main.py \
    --input_path samples/symphony.wav \
    --num_gaussians 2000 \
    --num_steps 15000 \
    --adaptive_add \
    --add_gaussians_steps 2000 4000 6000 8000 \
    --lr 0.005
```

## Advanced Training Techniques

### 1. Adaptive Gaussian Addition
Dynamically add Gaussians to high-error regions during training:

```bash
python main.py \
    --input_path samples/complex_audio.wav \
    --num_gaussians 500 \
    --adaptive_add \
    --add_gaussians_steps 1000 2000 3000 \
    --num_steps 5000
```

### 2. Loss Function Selection

- **L2 Loss**: General purpose audio
  ```bash
  --loss_type l2
  ```

- **Spectral Loss**: Music (preserves harmonic structure)
  ```bash
  --loss_type spectral --perceptual_weight 0.3
  ```

- **L1 Loss**: Speech (preserves clarity)
  ```bash
  --loss_type l1 --perceptual_weight 0.5
  ```

### 3. Quantization for Compression

Enable quantization after training to reduce file size:

```bash
# 8-bit quantization (high compression)
python main.py \
    --input_path samples/audio.wav \
    --num_gaussians 500 \
    --quantize \
    --bits_per_param 8

# 16-bit quantization (better quality)
python main.py \
    --input_path samples/audio.wav \
    --num_gaussians 500 \
    --quantize \
    --bits_per_param 16
```

## Batch Training Script

Create `batch_train.py`:

```python
import os
import subprocess
from pathlib import Path

# Configuration
SAMPLE_DIR = "samples"
OUTPUT_DIR = "outputs"
CONFIG_MAP = {
    "speech": "configs/speech.yaml",
    "music": "configs/music.yaml",
    "default": "configs/default.yaml"
}

def detect_audio_type(filename):
    """Detect audio type from filename"""
    name_lower = filename.lower()
    if any(word in name_lower for word in ["speech", "voice", "talk"]):
        return "speech"
    elif any(word in name_lower for word in ["music", "song", "piano", "guitar"]):
        return "music"
    return "default"

def train_audio(audio_path, audio_type="default"):
    """Train single audio file"""
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
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all audio files
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(Path(SAMPLE_DIR).glob(ext))

    print(f"Found {len(audio_files)} audio files")

    # Train each file
    for audio_file in audio_files:
        audio_type = detect_audio_type(audio_file.name)
        train_audio(str(audio_file), audio_type)

    print("Batch training completed!")

if __name__ == "__main__":
    main()
```

## Monitoring Training Progress

### View Logs
Training logs are saved in the `log_dir` directory:
```
logs/your_audio_g500_peaks/
├── log_train.txt          # Training log
├── cfg_train.yaml         # Training configuration
├── train/                 # Training outputs
│   ├── reconstructed.wav  # Reconstructed audio
│   └── gaussians_visualization.png  # Visualization
└── checkpoints/           # Model checkpoints
    ├── checkpoint_001000.pth
    ├── checkpoint_002000.pth
    └── ...
```

### Real-time Monitoring
```bash
# View training log
tail -f logs/your_audio_g500_peaks/log_train.txt

# Use tensorboard (if tensorboard support is added)
tensorboard --logdir logs/
```

## Model Evaluation

### Reconstruct Audio from Checkpoint
```bash
python main.py \
    --eval \
    --ckpt_file logs/your_audio_g500_peaks/checkpoints/checkpoint_005000.pth \
    --log_dir eval_output
```

### Compression Statistics
```
Original size: 1720.32 KB
Compressed size: 23.44 KB
Compression ratio: 73.38x
Bitrate: 12.3 kbps
SNR: 24.56 dB
```

## Common Issues

### 1. How to Choose Number of Gaussians?
- **Speech**: 200-500 Gaussians
- **Simple Music**: 500-1000 Gaussians
- **Complex Music**: 1000-3000 Gaussians
- **High Fidelity**: 3000+ Gaussians

### 2. Training Not Converging?
- Reduce learning rate: `--lr 0.001`
- Increase training steps: `--num_steps 10000`
- Try different initialization: `--init_method uniform`

### 3. Poor Reconstruction Quality?
- Increase number of Gaussians
- Enable adaptive addition: `--adaptive_add`
- Adjust perceptual weight: `--perceptual_weight 0.5`

### 4. File Too Large?
- Enable quantization: `--quantize --bits_per_param 6`
- Reduce number of Gaussians
- Use more aggressive quantization

## Performance Benchmarks

| Audio Type | Gaussians | Compression | SNR | Bitrate |
|------------|-----------|-------------|-----|---------|
| Speech | 300 | 100x | 20dB | 8kbps |
| Piano | 500 | 60x | 25dB | 16kbps |
| Pop Music | 1000 | 30x | 22dB | 32kbps |
| Orchestra | 2000 | 15x | 20dB | 64kbps |

## Next Steps

1. **Optimize Quality**: Adjust loss weights and Gaussian count
2. **Reduce File Size**: Enable quantization and entropy coding
3. **Batch Processing**: Use batch script for multiple files
4. **Real-time Encoding**: Develop streaming encoder