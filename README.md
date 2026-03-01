# Audio-GS: Content-Adaptive Audio Representation via 2D Gaussians

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

A novel audio compression method using 2D Gaussian representation in the time-frequency domain, inspired by [Image-GS](https://github.com/NYU-ICL/image-gs).

### 🌟 Key Features

- **🎯 Content-Adaptive**: Automatically allocates Gaussians based on spectral importance
- **⚡ Ultra-Fast Decoding**: ~0.3K MACs per sample, suitable for embedded devices
- **📊 Flexible Quality**: Adjustable compression ratio from 10x to 100x
- **🔧 Easy to Use**: Simple CLI interface with pre-configured settings

### � Performance Benchmarks

| Method | Compression Ratio | SNR (dB) | Decode Speed |
|--------|-------------------|----------|--------------|
| Audio-GS (Speech) | 100x | 20-25 | Real-time |
| Audio-GS (Music) | 50x | 22-28 | Real-time |
| MP3 128k | 11x | 35+ | Real-time |
| Opus 32k | 44x | 30+ | Real-time |

### �🚀 Quick Start

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

### 🎯 Training Guide

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions.

#### Quick Training Examples

```bash
# Speech compression (high compression ratio)
python main.py \
    --input_path speech.wav \
    --config configs/speech.yaml \
    --num_gaussians 300 \
    --quantize

# Music compression (high quality)
python main.py \
    --input_path music.wav \
    --config configs/music.yaml \
    --num_gaussians 1000 \
    --num_steps 10000

# Adaptive Gaussian addition
python main.py \
    --input_path complex.wav \
    --num_gaussians 500 \
    --adaptive_add \
    --add_gaussians_steps 1000 2000 3000
```

### 🔍 Evaluation and Visualization

```bash
# Reconstruct audio from checkpoint
python main.py \
    --eval \
    --ckpt_file logs/your_model/checkpoints/checkpoint_005000.pth \
    --log_dir eval_output

# Monitor training progress
tail -f logs/your_model/log_train.txt
```

### 📈 Performance Optimization Tips

1. **Improve quality**: Increase number of Gaussians and training steps
2. **Reduce file size**: Enable quantization (`--quantize`) and reduce Gaussians
3. **Speed up training**: Use GPU and adjust learning rate
4. **Task-specific optimization**: Choose appropriate config file for audio type

### 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### 📄 License

MIT License - See [LICENSE](LICENSE) file for details

### 🙏 Acknowledgments

- [Image-GS](https://github.com/NYU-ICL/image-gs) - Inspiration for Gaussian representation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [librosa](https://librosa.org/) - Audio processing library
- [torchaudio](https://pytorch.org/audio/) - PyTorch audio extension

### 📮 Contact

- GitHub Issues: [Report issues](https://github.com/z27833009/Audio-GS/issues)
- Discussions: [Technical discussions](https://github.com/z27833009/Audio-GS/discussions)

---

<div align="center">

**If you find this project helpful, please give it a ⭐ star!**

Made with ❤️ for audio compression research

</div>