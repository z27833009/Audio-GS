import os
import shutil
import yaml
import torch
import numpy as np
import random


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_dir(path):
    """Clean and remove directory if exists"""
    if os.path.exists(path):
        shutil.rmtree(path)


def save_cfg(path, args):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def load_cfg(path):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def get_compression_ratio(original_size, compressed_size):
    """Calculate compression ratio"""
    return original_size / compressed_size


def get_bitrate(file_size_bytes, duration_seconds, unit='kbps'):
    """Calculate bitrate
    
    Args:
        file_size_bytes: File size in bytes
        duration_seconds: Duration in seconds
        unit: 'bps', 'kbps', or 'mbps'
    
    Returns:
        Bitrate in specified unit
    """
    bitrate_bps = (file_size_bytes * 8) / duration_seconds
    
    if unit == 'kbps':
        return bitrate_bps / 1000
    elif unit == 'mbps':
        return bitrate_bps / 1000000
    else:
        return bitrate_bps


def count_parameters(model):
    """Count number of parameters in model"""
    return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    """Format time in seconds to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"