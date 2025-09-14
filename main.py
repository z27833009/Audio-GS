import argparse
import os
import torch
import yaml
from model import AudioGaussian2D


def parse_args():
    parser = argparse.ArgumentParser(description='Audio-GS: Audio Compression via 2D Gaussians')
    
    # Input/Output
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs and outputs')
    parser.add_argument('--ckpt_file', type=str, default=None,
                        help='Checkpoint file for evaluation')
    
    # Model parameters
    parser.add_argument('--num_gaussians', type=int, default=500,
                        help='Number of Gaussians')
    parser.add_argument('--init_method', type=str, default='peaks',
                        choices=['peaks', 'random', 'uniform'],
                        help='Gaussian initialization method')
    
    # Audio parameters
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='STFT hop length')
    parser.add_argument('--win_length', type=int, default=2048,
                        help='STFT window length')
    
    # Optimization
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.995,
                        help='Learning rate decay')
    parser.add_argument('--optimizer_type', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='spectral',
                        choices=['l1', 'l2', 'spectral'],
                        help='Loss function type')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--phase_weight', type=float, default=0.0,
                        help='Weight for phase loss')
    
    # Adaptive refinement
    parser.add_argument('--adaptive_add', action='store_true',
                        help='Adaptively add Gaussians during training')
    parser.add_argument('--add_gaussians_steps', type=int, nargs='+',
                        default=[1000, 2000, 3000],
                        help='Steps to add Gaussians')
    
    # Quantization
    parser.add_argument('--quantize', action='store_true',
                        help='Enable quantization')
    parser.add_argument('--bits_per_param', type=int, default=8,
                        help='Bits per parameter for quantization')
    
    # Logging
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level')
    
    # Mode
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation mode')
    
    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (overrides command line args)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            setattr(args, key, value)
    
    # Create log directory name based on parameters
    if not args.eval and not os.path.exists(args.log_dir):
        audio_name = os.path.splitext(os.path.basename(args.input_path))[0]
        args.log_dir = os.path.join(
            args.log_dir,
            f"{audio_name}_g{args.num_gaussians}_{args.init_method}"
        )
    
    return args


def main():
    args = parse_args()
    
    print(f"Audio-GS: {'Evaluation' if args.eval else 'Training'} mode")
    print(f"Device: {args.device}")
    print(f"Input: {args.input_path}")
    print(f"Gaussians: {args.num_gaussians}")
    
    # Create model
    model = AudioGaussian2D(args)
    
    if args.eval:
        # Evaluation mode - decode and save audio
        print("\nDecoding audio from Gaussians...")
        with torch.no_grad():
            rendered_magnitude, rendered_complex = model.forward()
            
            # Convert to audio
            from utils.audio_utils import stft_to_audio, save_audio, compute_snr
            reconstructed_audio = stft_to_audio(
                rendered_complex,
                model.hop_length,
                model.win_length
            )
            
            # Save reconstructed audio
            output_path = os.path.join(model.eval_dir, 'reconstructed.wav')
            save_audio(output_path, reconstructed_audio.cpu(), model.sample_rate)
            
            # Compute metrics
            if hasattr(model, 'audio'):
                snr = compute_snr(model.audio, reconstructed_audio)
                print(f"SNR: {snr:.2f} dB")
            
            print(f"Saved reconstructed audio to {output_path}")
            
            # Visualize
            from utils.audio_utils import visualize_gaussians_on_spectrogram
            viz_path = os.path.join(model.eval_dir, 'gaussians_visualization.png')
            visualize_gaussians_on_spectrogram(
                model.target_magnitude.cpu(),
                model.time_centers.detach().cpu(),
                model.freq_centers.detach().cpu(),
                model.time_spreads.detach().cpu(),
                model.freq_spreads.detach().cpu(),
                save_path=viz_path
            )
            print(f"Saved visualization to {viz_path}")
    else:
        # Training mode - optimize Gaussians
        print("\nStarting optimization...")
        model.optimize()
        
        # Print compression statistics
        from utils.quantization_utils import estimate_compressed_size
        
        original_size = model.audio.numel() * 2  # 16-bit audio
        compressed_size = estimate_compressed_size(
            model.num_gaussians,
            args.bits_per_param if args.quantize else 32,
            use_entropy_coding=True
        )
        
        compression_ratio = original_size / compressed_size
        print(f"\nCompression Statistics:")
        print(f"Original size: {original_size / 1024:.2f} KB")
        print(f"Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Calculate bitrate
        duration = len(model.audio) / model.sample_rate
        bitrate = (compressed_size * 8) / duration / 1000  # kbps
        print(f"Bitrate: {bitrate:.1f} kbps")


if __name__ == '__main__':
    main()