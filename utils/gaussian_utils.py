import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks_2d
from scipy.ndimage import maximum_filter


def gaussian_2d(x, y, mx, my, sx, sy):
    """Compute 2D Gaussian value at position (x, y)"""
    return torch.exp(-0.5 * (((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2))


def render_spectrogram_from_gaussians(
    time_centers,
    freq_centers,
    time_spreads,
    freq_spreads,
    magnitudes,
    shape,
    device='cuda'
):
    """Render spectrogram from 2D Gaussians
    
    Args:
        time_centers: (N,) tensor of time centers
        freq_centers: (N,) tensor of frequency centers
        time_spreads: (N,) tensor of time spreads (std dev)
        freq_spreads: (N,) tensor of frequency spreads (std dev)
        magnitudes: (N,) tensor of magnitudes
        shape: (H, W) output shape (freq_bins, time_frames)
        device: device to render on
    
    Returns:
        Rendered spectrogram of shape (H, W)
    """
    H, W = shape
    N = len(time_centers)
    
    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Initialize output
    spectrogram = torch.zeros((H, W), device=device)
    
    # Render each Gaussian
    for i in range(N):
        # Skip Gaussians that are too far from the image
        if (time_centers[i] < -3 * time_spreads[i] or 
            time_centers[i] > W + 3 * time_spreads[i] or
            freq_centers[i] < -3 * freq_spreads[i] or 
            freq_centers[i] > H + 3 * freq_spreads[i]):
            continue
        
        # Compute Gaussian contribution
        gaussian = gaussian_2d(
            X, Y,
            time_centers[i], freq_centers[i],
            time_spreads[i], freq_spreads[i]
        )
        
        # Add weighted contribution
        spectrogram += magnitudes[i] * gaussian
    
    return spectrogram


def render_spectrogram_batch(
    gaussians_batch,
    shape,
    device='cuda'
):
    """Render batch of spectrograms from Gaussians
    
    Args:
        gaussians_batch: dict with batched parameters
        shape: (H, W) output shape
        device: device to render on
    
    Returns:
        Batch of rendered spectrograms
    """
    batch_size = gaussians_batch['time_centers'].shape[0]
    H, W = shape
    
    spectrograms = []
    for b in range(batch_size):
        spec = render_spectrogram_from_gaussians(
            gaussians_batch['time_centers'][b],
            gaussians_batch['freq_centers'][b],
            gaussians_batch['time_spreads'][b],
            gaussians_batch['freq_spreads'][b],
            gaussians_batch['magnitudes'][b],
            shape,
            device
        )
        spectrograms.append(spec)
    
    return torch.stack(spectrograms)


def initialize_gaussians_from_peaks(
    spectrogram,
    num_gaussians,
    device='cuda',
    min_distance=5
):
    """Initialize Gaussians at spectrogram peaks
    
    Args:
        spectrogram: Input spectrogram tensor
        num_gaussians: Number of Gaussians to initialize
        device: Device to create tensors on
        min_distance: Minimum distance between peaks
    
    Returns:
        Dictionary of Gaussian parameters
    """
    if isinstance(spectrogram, torch.Tensor):
        spec_np = spectrogram.cpu().numpy()
    else:
        spec_np = spectrogram
    
    # Find local maxima
    local_max = maximum_filter(spec_np, size=min_distance)
    peaks_mask = (spec_np == local_max) & (spec_np > np.percentile(spec_np, 75))
    
    # Get peak coordinates
    peak_coords = np.argwhere(peaks_mask)
    peak_values = spec_np[peaks_mask]
    
    # Sort by magnitude and select top-k
    sorted_indices = np.argsort(peak_values)[::-1]
    n_peaks = min(num_gaussians, len(sorted_indices))
    selected_indices = sorted_indices[:n_peaks]
    
    selected_coords = peak_coords[selected_indices]
    selected_values = peak_values[selected_indices]
    
    # Initialize Gaussian parameters
    time_centers = torch.tensor(
        selected_coords[:, 1].astype(np.float32),
        device=device
    )
    freq_centers = torch.tensor(
        selected_coords[:, 0].astype(np.float32),
        device=device
    )
    
    # Estimate spreads based on local energy distribution
    time_spreads = torch.ones(n_peaks, device=device) * 3.0
    freq_spreads = torch.ones(n_peaks, device=device) * 5.0
    
    # Initialize magnitudes based on peak values
    magnitudes = torch.tensor(selected_values, device=device, dtype=torch.float32)
    magnitudes = magnitudes / magnitudes.max()  # Normalize
    
    # Initialize phases (for complex representation)
    phases = torch.zeros(n_peaks, device=device)
    
    # If we need more Gaussians, add random ones
    if n_peaks < num_gaussians:
        n_random = num_gaussians - n_peaks
        H, W = spec_np.shape
        
        random_time = torch.rand(n_random, device=device) * W
        random_freq = torch.rand(n_random, device=device) * H
        random_time_spreads = torch.ones(n_random, device=device) * 3.0
        random_freq_spreads = torch.ones(n_random, device=device) * 5.0
        random_mags = torch.ones(n_random, device=device) * 0.1
        random_phases = torch.zeros(n_random, device=device)
        
        time_centers = torch.cat([time_centers, random_time])
        freq_centers = torch.cat([freq_centers, random_freq])
        time_spreads = torch.cat([time_spreads, random_time_spreads])
        freq_spreads = torch.cat([freq_spreads, random_freq_spreads])
        magnitudes = torch.cat([magnitudes, random_mags])
        phases = torch.cat([phases, random_phases])
    
    return {
        'time_centers': time_centers,
        'freq_centers': freq_centers,
        'time_spreads': time_spreads,
        'freq_spreads': freq_spreads,
        'magnitudes': magnitudes,
        'phases': phases
    }


def adaptive_gaussian_refinement(
    error_map,
    current_gaussians,
    n_add=10,
    n_remove=5,
    device='cuda'
):
    """Adaptively refine Gaussians based on reconstruction error
    
    Args:
        error_map: Reconstruction error map
        current_gaussians: Current Gaussian parameters
        n_add: Number of Gaussians to add
        n_remove: Number of Gaussians to remove
        device: Device
    
    Returns:
        Refined Gaussian parameters
    """
    # Find high error regions for adding Gaussians
    if isinstance(error_map, torch.Tensor):
        error_np = error_map.cpu().numpy()
    else:
        error_np = error_map
    
    # Find peaks in error map
    local_max = maximum_filter(error_np, size=3)
    peaks_mask = (error_np == local_max) & (error_np > np.percentile(error_np, 90))
    peak_coords = np.argwhere(peaks_mask)
    
    # Add new Gaussians at error peaks
    if len(peak_coords) > 0 and n_add > 0:
        n_add_actual = min(n_add, len(peak_coords))
        selected_indices = np.random.choice(len(peak_coords), n_add_actual, replace=False)
        selected_coords = peak_coords[selected_indices]
        
        new_time = torch.tensor(selected_coords[:, 1].astype(np.float32), device=device)
        new_freq = torch.tensor(selected_coords[:, 0].astype(np.float32), device=device)
        new_time_spreads = torch.ones(n_add_actual, device=device) * 2.0
        new_freq_spreads = torch.ones(n_add_actual, device=device) * 3.0
        new_mags = torch.ones(n_add_actual, device=device) * 0.5
        new_phases = torch.zeros(n_add_actual, device=device)
        
        # Concatenate with existing
        current_gaussians['time_centers'] = torch.cat([
            current_gaussians['time_centers'], new_time
        ])
        current_gaussians['freq_centers'] = torch.cat([
            current_gaussians['freq_centers'], new_freq
        ])
        current_gaussians['time_spreads'] = torch.cat([
            current_gaussians['time_spreads'], new_time_spreads
        ])
        current_gaussians['freq_spreads'] = torch.cat([
            current_gaussians['freq_spreads'], new_freq_spreads
        ])
        current_gaussians['magnitudes'] = torch.cat([
            current_gaussians['magnitudes'], new_mags
        ])
        current_gaussians['phases'] = torch.cat([
            current_gaussians['phases'], new_phases
        ])
    
    # Remove low-contribution Gaussians
    if n_remove > 0 and len(current_gaussians['magnitudes']) > n_remove:
        # Keep Gaussians with highest magnitudes
        keep_indices = torch.argsort(current_gaussians['magnitudes'], descending=True)[:-n_remove]
        
        for key in current_gaussians:
            current_gaussians[key] = current_gaussians[key][keep_indices]
    
    return current_gaussians