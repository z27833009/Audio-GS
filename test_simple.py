"""Simple test script to verify Audio-GS installation and basic functionality"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.gaussian_utils import render_spectrogram_from_gaussians, initialize_gaussians_from_peaks
from utils.audio_utils import compute_snr

def test_gaussian_rendering():
    """Test basic Gaussian rendering"""
    print("Testing Gaussian rendering...")

    # Create test parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = (256, 100)  # freq_bins, time_frames
    num_gaussians = 10

    # Random Gaussian parameters
    time_centers = torch.rand(num_gaussians, device=device) * shape[1]
    freq_centers = torch.rand(num_gaussians, device=device) * shape[0]
    time_spreads = torch.ones(num_gaussians, device=device) * 5.0
    freq_spreads = torch.ones(num_gaussians, device=device) * 10.0
    magnitudes = torch.rand(num_gaussians, device=device)

    # Render spectrogram
    spectrogram = render_spectrogram_from_gaussians(
        time_centers, freq_centers,
        time_spreads, freq_spreads,
        magnitudes, shape, device
    )

    assert spectrogram.shape == shape
    assert torch.all(spectrogram >= 0)
    print(f"✓ Rendered spectrogram shape: {spectrogram.shape}")
    print(f"✓ Spectrogram range: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]")

    return spectrogram

def test_peak_initialization():
    """Test Gaussian initialization from peaks"""
    print("\nTesting peak-based initialization...")

    # Create synthetic spectrogram with clear peaks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = (128, 50)
    spectrogram = torch.zeros(shape, device=device)

    # Add some peaks
    peak_positions = [(30, 10), (60, 25), (90, 40)]
    for freq, time in peak_positions:
        spectrogram[freq, time] = 1.0
        # Add some spread
        for df in range(-2, 3):
            for dt in range(-2, 3):
                if 0 <= freq+df < shape[0] and 0 <= time+dt < shape[1]:
                    spectrogram[freq+df, time+dt] += 0.5 * np.exp(-(df**2 + dt**2)/4)

    # Initialize Gaussians from peaks
    gaussians = initialize_gaussians_from_peaks(
        spectrogram,
        num_gaussians=5,
        device=device
    )

    assert len(gaussians['time_centers']) == 5
    print(f"✓ Initialized {len(gaussians['time_centers'])} Gaussians")
    print(f"✓ Time centers range: [{gaussians['time_centers'].min():.1f}, {gaussians['time_centers'].max():.1f}]")
    print(f"✓ Freq centers range: [{gaussians['freq_centers'].min():.1f}, {gaussians['freq_centers'].max():.1f}]")

def test_snr_computation():
    """Test SNR computation"""
    print("\nTesting SNR computation...")

    # Create test signals
    original = torch.randn(44100)  # 1 second at 44.1kHz
    noise = torch.randn_like(original) * 0.1
    reconstructed = original + noise

    snr = compute_snr(original, reconstructed)
    print(f"✓ SNR: {snr:.2f} dB")

    # Perfect reconstruction should have infinite SNR
    perfect_snr = compute_snr(original, original)
    assert perfect_snr > 100 or perfect_snr == float('inf')
    print(f"✓ Perfect reconstruction SNR: {perfect_snr:.2f} dB")

def visualize_test_results():
    """Visualize test Gaussian rendering"""
    print("\nGenerating visualization...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = (128, 200)
    num_gaussians = 15

    # Create interesting Gaussian pattern
    time_centers = torch.tensor([20, 40, 60, 80, 100, 120, 140, 160, 180,
                                 50, 100, 150, 75, 125, 175], device=device, dtype=torch.float32)
    freq_centers = torch.tensor([20, 40, 60, 80, 100, 80, 60, 40, 20,
                                 64, 64, 64, 32, 96, 50], device=device, dtype=torch.float32)
    time_spreads = torch.ones(num_gaussians, device=device) * 8.0
    freq_spreads = torch.ones(num_gaussians, device=device) * 12.0
    magnitudes = torch.rand(num_gaussians, device=device) * 0.8 + 0.2

    # Render
    spectrogram = render_spectrogram_from_gaussians(
        time_centers, freq_centers,
        time_spreads, freq_spreads,
        magnitudes, shape, device
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Test Gaussian Spectrogram Rendering')

    # Overlay Gaussian centers
    plt.scatter(time_centers.cpu(), freq_centers.cpu(), c='red', s=20, marker='x', label='Gaussian Centers')
    plt.legend()

    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=150)
    print(f"✓ Saved visualization to test_visualization.png")
    plt.close()

def main():
    print("="*50)
    print("Audio-GS Test Suite")
    print("="*50)

    try:
        # Run tests
        test_gaussian_rendering()
        test_peak_initialization()
        test_snr_computation()
        visualize_test_results()

        print("\n" + "="*50)
        print("All tests passed successfully! ✓")
        print("Audio-GS is ready to use.")
        print("="*50)

        print("\nNext steps:")
        print("1. Place audio files in the 'samples/' directory")
        print("2. Run: python main.py --input_path samples/your_audio.wav --num_gaussians 500")
        print("3. Check the 'logs/' directory for results")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())