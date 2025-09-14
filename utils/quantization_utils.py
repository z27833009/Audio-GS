import torch
import numpy as np


def quantize_gaussians(checkpoint, bits=8):
    """Quantize Gaussian parameters to reduce storage
    
    Args:
        checkpoint: Model checkpoint with Gaussian parameters
        bits: Number of bits per parameter
    
    Returns:
        Quantized checkpoint
    """
    quantized = checkpoint.copy()
    
    # Parameters to quantize
    param_keys = [
        'time_centers', 'freq_centers',
        'time_spreads', 'freq_spreads',
        'magnitudes', 'phases'
    ]
    
    for key in param_keys:
        if key not in checkpoint:
            continue
            
        param = checkpoint[key]
        
        # Get min and max for scaling
        param_min = param.min().item()
        param_max = param.max().item()
        
        # Scale to [0, 2^bits - 1]
        scale = (2 ** bits - 1) / (param_max - param_min + 1e-8)
        offset = param_min
        
        # Quantize
        quantized_param = torch.round((param - offset) * scale).to(torch.uint8 if bits <= 8 else torch.int16)
        
        # Store quantized parameters and metadata
        quantized[f'{key}_quantized'] = quantized_param
        quantized[f'{key}_scale'] = scale
        quantized[f'{key}_offset'] = offset
        
        # Remove original parameter
        del quantized[key]
    
    quantized['quantization_bits'] = bits
    return quantized


def dequantize_gaussians(quantized_checkpoint):
    """Dequantize Gaussian parameters
    
    Args:
        quantized_checkpoint: Quantized checkpoint
    
    Returns:
        Dequantized checkpoint
    """
    checkpoint = quantized_checkpoint.copy()
    
    # Parameters to dequantize
    param_keys = [
        'time_centers', 'freq_centers',
        'time_spreads', 'freq_spreads',
        'magnitudes', 'phases'
    ]
    
    for key in param_keys:
        quantized_key = f'{key}_quantized'
        if quantized_key not in quantized_checkpoint:
            continue
        
        # Get quantized data and metadata
        quantized_param = quantized_checkpoint[quantized_key].float()
        scale = quantized_checkpoint[f'{key}_scale']
        offset = quantized_checkpoint[f'{key}_offset']
        
        # Dequantize
        param = (quantized_param / scale) + offset
        
        # Store dequantized parameter
        checkpoint[key] = param
        
        # Remove quantized data and metadata
        for suffix in ['_quantized', '_scale', '_offset']:
            k = f'{key}{suffix}'
            if k in checkpoint:
                del checkpoint[k]
    
    if 'quantization_bits' in checkpoint:
        del checkpoint['quantization_bits']
    
    return checkpoint


def estimate_compressed_size(num_gaussians, bits_per_param=8, use_entropy_coding=True):
    """Estimate compressed size in bytes
    
    Args:
        num_gaussians: Number of Gaussians
        bits_per_param: Bits per parameter
        use_entropy_coding: Whether entropy coding will be used
    
    Returns:
        Estimated size in bytes
    """
    # 6 parameters per Gaussian (time, freq, spreads, magnitude, phase)
    params_per_gaussian = 6
    
    # Base size
    bits_total = num_gaussians * params_per_gaussian * bits_per_param
    bytes_total = bits_total / 8
    
    # Entropy coding typically achieves ~60-70% of original size
    if use_entropy_coding:
        bytes_total *= 0.65
    
    # Add metadata overhead (~1KB)
    bytes_total += 1024
    
    return int(bytes_total)


def adaptive_quantization(gaussians, importance_scores, bit_budget):
    """Adaptive quantization based on Gaussian importance
    
    Args:
        gaussians: Gaussian parameters
        importance_scores: Importance score for each Gaussian
        bit_budget: Total bit budget
    
    Returns:
        Adaptively quantized Gaussians
    """
    num_gaussians = len(importance_scores)
    
    # Sort by importance
    sorted_indices = torch.argsort(importance_scores, descending=True)
    
    # Allocate bits based on importance
    # Top 20% get 16 bits, next 30% get 8 bits, rest get 4 bits
    n_high = int(0.2 * num_gaussians)
    n_medium = int(0.3 * num_gaussians)
    
    bit_allocation = torch.zeros(num_gaussians, dtype=torch.int)
    bit_allocation[sorted_indices[:n_high]] = 16
    bit_allocation[sorted_indices[n_high:n_high+n_medium]] = 8
    bit_allocation[sorted_indices[n_high+n_medium:]] = 4
    
    # Quantize each Gaussian with its allocated bits
    quantized_gaussians = {}
    for key in gaussians:
        quantized_gaussians[key] = []
        for i, bits in enumerate(bit_allocation):
            # Quantize individual Gaussian
            param_value = gaussians[key][i]
            # ... quantization logic based on bits ...
            quantized_gaussians[key].append(param_value)  # Placeholder
    
    return quantized_gaussians, bit_allocation