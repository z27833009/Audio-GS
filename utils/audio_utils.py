import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def load_audio(path, sr=44100, mono=True):
    """Load audio file"""
    waveform, sample_rate = torchaudio.load(path)
    
    # Resample if needed
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)
    
    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.squeeze(0), sr


def save_audio(path, waveform, sr=44100):
    """Save audio to file"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    sf.write(path, waveform, sr)


def stft_to_audio(stft_complex, hop_length, win_length, n_iter=32):
    """Convert complex STFT back to audio using Griffin-Lim"""
    if isinstance(stft_complex, torch.Tensor):
        # Use torchaudio's inverse STFT
        istft_transform = torchaudio.transforms.InverseSpectrogram(
            n_fft=2 * (stft_complex.shape[-2] - 1),
            hop_length=hop_length,
            win_length=win_length
        )
        audio = istft_transform(stft_complex)
    else:
        # Use librosa
        audio = librosa.istft(
            stft_complex.cpu().numpy(),
            hop_length=hop_length,
            win_length=win_length
        )
        audio = torch.from_numpy(audio)
    
    return audio


def griffin_lim(magnitude, n_iter=32, hop_length=512, win_length=2048):
    """Griffin-Lim algorithm for phase reconstruction"""
    if isinstance(magnitude, torch.Tensor):
        magnitude = magnitude.cpu().numpy()
    
    audio = librosa.griffinlim(
        magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length
    )
    
    return torch.from_numpy(audio)


def compute_snr(original, reconstructed):
    """Compute Signal-to-Noise Ratio in dB"""
    if len(original.shape) > 1:
        original = original.squeeze()
    if len(reconstructed.shape) > 1:
        reconstructed = reconstructed.squeeze()
    
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_spectral_loss(pred_mag, target_mag, alpha=0.5):
    """Compute spectral convergence and log magnitude loss"""
    # Spectral convergence
    sc = torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')
    
    # Log magnitude loss
    eps = 1e-8
    log_mag_loss = torch.mean(torch.abs(
        torch.log(target_mag + eps) - torch.log(pred_mag + eps)
    ))
    
    return alpha * sc + (1 - alpha) * log_mag_loss


def get_mel_spectrogram(audio, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """Compute mel-spectrogram"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    if audio.device != mel_transform.device:
        mel_transform = mel_transform.to(audio.device)
    
    return mel_transform(audio)


def visualize_gaussians_on_spectrogram(
    spectrogram,
    time_centers,
    freq_centers,
    time_spreads,
    freq_spreads,
    save_path=None,
    top_k=50
):
    """Visualize Gaussians overlaid on spectrogram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Convert to numpy if needed
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()
    if isinstance(time_centers, torch.Tensor):
        time_centers = time_centers.cpu().numpy()
    if isinstance(freq_centers, torch.Tensor):
        freq_centers = freq_centers.cpu().numpy()
    if isinstance(time_spreads, torch.Tensor):
        time_spreads = time_spreads.cpu().numpy()
    if isinstance(freq_spreads, torch.Tensor):
        freq_spreads = freq_spreads.cpu().numpy()
    
    # Plot original spectrogram
    ax1.imshow(
        20 * np.log10(spectrogram + 1e-8),
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    ax1.set_title('Original Spectrogram')
    ax1.set_xlabel('Time Frame')
    ax1.set_ylabel('Frequency Bin')
    
    # Plot spectrogram with Gaussians
    ax2.imshow(
        20 * np.log10(spectrogram + 1e-8),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        alpha=0.5
    )
    
    # Draw top-k Gaussians as ellipses
    magnitudes = time_spreads * freq_spreads  # Use area as proxy for importance
    top_indices = np.argsort(magnitudes)[-top_k:]
    
    for idx in top_indices:
        ellipse = Ellipse(
            (time_centers[idx], freq_centers[idx]),
            width=2 * time_spreads[idx],
            height=2 * freq_spreads[idx],
            fill=False,
            edgecolor='red',
            linewidth=1,
            alpha=0.7
        )
        ax2.add_patch(ellipse)
    
    ax2.set_title(f'Spectrogram with Top {top_k} Gaussians')
    ax2.set_xlabel('Time Frame')
    ax2.set_ylabel('Frequency Bin')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig