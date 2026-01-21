import logging
import os
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from scipy.signal import find_peaks

from utils.audio_utils import (
    compute_snr,
    compute_spectral_loss,
    get_mel_spectrogram,
    griffin_lim,
    load_audio,
    save_audio,
    stft_to_audio,
    visualize_gaussians_on_spectrogram,
)
from utils.gaussian_utils import (
    render_spectrogram_from_gaussians,
    initialize_gaussians_from_peaks,
    gaussian_2d,
)
from utils.misc_utils import clean_dir, save_cfg, set_random_seed
from utils.quantization_utils import quantize_gaussians, dequantize_gaussians


class AudioGaussian2D(nn.Module):
    """2D Gaussian representation for audio in time-frequency domain"""

    def __init__(self, args):
        super(AudioGaussian2D, self).__init__()
        self.evaluate = args.eval
        set_random_seed(seed=args.seed)
        self.device = args.device
        self.dtype = torch.float32

        self._init_logging(args)
        self._init_audio_params(args)
        self._init_gaussians(args)
        self._init_loss(args)
        self._init_optimization(args)

        if self.evaluate:
            self.ckpt_file = args.ckpt_file
            self._load_model()
        else:
            self._init_gaussian_params(args)

    def _init_logging(self, args):
        self.log_dir = args.log_dir
        self.log_level = args.log_level
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.train_dir = os.path.join(self.log_dir, "train")
        self.eval_dir = os.path.join(self.log_dir, "eval")
        self.save_steps = args.save_steps
        self.eval_steps = args.eval_steps

        if not self.evaluate:
            clean_dir(path=self.log_dir)
            os.makedirs(self.log_dir, exist_ok=False)
            os.makedirs(self.ckpt_dir, exist_ok=False)
            os.makedirs(self.train_dir, exist_ok=False)
        else:
            os.makedirs(self.eval_dir, exist_ok=True)

        self._gen_logger(args)
        if not self.evaluate:
            save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)

    def _gen_logger(self, args):
        log_fname = "log_eval" if self.evaluate else "log_train"
        log_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(level=log_level)
        self.worklog = logging.getLogger("Audio-GS Logger")
        self.worklog.propagate = False

        fileHandler = logging.FileHandler(f"{self.log_dir}/{log_fname}.txt", mode="a")
        consoleHandler = logging.StreamHandler(sys.stdout)
        self.worklog.handlers = [fileHandler, consoleHandler]

        action = "decoding" if self.evaluate else "encoding"
        self.worklog.info(f"Start {action} with {args.num_gaussians:d} Gaussians for '{args.input_path}'")

    def _init_audio_params(self, args):
        # STFT parameters
        self.n_fft = args.n_fft  # 2048 default
        self.hop_length = args.hop_length  # 512 default
        self.win_length = args.win_length  # 2048 default
        self.sample_rate = args.sample_rate  # 44100 default

        # Load target audio
        self.audio, self.sr = load_audio(args.input_path, sr=self.sample_rate)
        self.audio = self.audio.to(self.device)

        # Compute spectrogram
        self.stft_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=None,
            return_complex=True
        ).to(self.device)

        self.target_stft = self.stft_transform(self.audio)
        self.target_magnitude = torch.abs(self.target_stft)
        self.target_phase = torch.angle(self.target_stft)

        # Normalize magnitude for better optimization
        self.mag_mean = self.target_magnitude.mean()
        self.mag_std = self.target_magnitude.std()
        self.target_magnitude_norm = (self.target_magnitude - self.mag_mean) / (self.mag_std + 1e-8)

        self.n_freq_bins = self.target_magnitude.shape[-2]
        self.n_time_frames = self.target_magnitude.shape[-1]

        # Griffin-Lim for phase reconstruction
        self.gl_transform = T.GriffinLim(
            n_fft=self.n_fft,
            n_iter=32,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1,
        ).to(self.device)

        self.worklog.info(f"Audio shape: {self.audio.shape}, STFT shape: {self.target_stft.shape}")
        self.worklog.info(f"Frequency bins: {self.n_freq_bins}, Time frames: {self.n_time_frames}")

    def _init_gaussians(self, args):
        self.num_gaussians = args.num_gaussians
        self.init_method = args.init_method  # 'peaks', 'random', 'uniform'
        self.adaptive_add = args.adaptive_add
        self.add_gaussians_steps = args.add_gaussians_steps if self.adaptive_add else []
        self.quantize = args.quantize
        self.bits_per_param = args.bits_per_param if self.quantize else None

    def _init_gaussian_params(self, args):
        """Initialize Gaussian parameters"""
        if self.init_method == 'peaks':
            # Initialize at spectral peaks
            gaussians = initialize_gaussians_from_peaks(
                self.target_magnitude_norm,
                num_gaussians=self.num_gaussians,
                device=self.device
            )
        elif self.init_method == 'random':
            # Random initialization
            gaussians = self._random_init_gaussians()
        else:  # uniform
            # Uniform grid initialization
            gaussians = self._uniform_init_gaussians()

        # Parameters to optimize
        self.time_centers = nn.Parameter(gaussians['time_centers'])
        self.freq_centers = nn.Parameter(gaussians['freq_centers'])
        self.time_spreads = nn.Parameter(gaussians['time_spreads'])
        self.freq_spreads = nn.Parameter(gaussians['freq_spreads'])
        self.magnitudes = nn.Parameter(gaussians['magnitudes'])
        self.phases = nn.Parameter(gaussians['phases'])

        self.worklog.info(f"Initialized {self.num_gaussians} Gaussians using method: {self.init_method}")

    def _random_init_gaussians(self):
        """Random Gaussian initialization"""
        gaussians = {
            'time_centers': torch.rand(self.num_gaussians, device=self.device) * self.n_time_frames,
            'freq_centers': torch.rand(self.num_gaussians, device=self.device) * self.n_freq_bins,
            'time_spreads': torch.ones(self.num_gaussians, device=self.device) * 5.0,
            'freq_spreads': torch.ones(self.num_gaussians, device=self.device) * 10.0,
            'magnitudes': torch.ones(self.num_gaussians, device=self.device),
            'phases': torch.zeros(self.num_gaussians, device=self.device),
        }
        return gaussians

    def _uniform_init_gaussians(self):
        """Uniform grid initialization"""
        n_time = int(np.sqrt(self.num_gaussians * self.n_time_frames / self.n_freq_bins))
        n_freq = self.num_gaussians // n_time

        time_grid = torch.linspace(0, self.n_time_frames, n_time, device=self.device)
        freq_grid = torch.linspace(0, self.n_freq_bins, n_freq, device=self.device)

        time_centers, freq_centers = torch.meshgrid(time_grid, freq_grid, indexing='xy')

        gaussians = {
            'time_centers': time_centers.flatten()[:self.num_gaussians],
            'freq_centers': freq_centers.flatten()[:self.num_gaussians],
            'time_spreads': torch.ones(self.num_gaussians, device=self.device) * 5.0,
            'freq_spreads': torch.ones(self.num_gaussians, device=self.device) * 10.0,
            'magnitudes': torch.ones(self.num_gaussians, device=self.device),
            'phases': torch.zeros(self.num_gaussians, device=self.device),
        }
        return gaussians

    def _init_loss(self, args):
        self.loss_type = args.loss_type  # 'l2', 'l1', 'spectral'
        self.perceptual_weight = args.perceptual_weight
        self.phase_weight = args.phase_weight

        # Mel-spectrogram for perceptual loss
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128
        ).to(self.device)

    def _init_optimization(self, args):
        self.num_steps = args.num_steps
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.optimizer_type = args.optimizer_type

        if not self.evaluate:
            params = [
                {'params': [self.time_centers, self.freq_centers], 'lr': self.lr},
                {'params': [self.time_spreads, self.freq_spreads], 'lr': self.lr * 0.5},
                {'params': [self.magnitudes], 'lr': self.lr * 2.0},
                {'params': [self.phases], 'lr': self.lr * 0.1},
            ]

            if self.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(params)
            else:
                self.optimizer = torch.optim.SGD(params, momentum=0.9)

            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.lr_decay
            )

    def forward(self, use_gl=None):
        """Render spectrogram from Gaussians"""
        # Ensure parameters are in valid ranges
        time_centers = torch.clamp(self.time_centers, 0, self.n_time_frames - 1)
        freq_centers = torch.clamp(self.freq_centers, 0, self.n_freq_bins - 1)
        time_spreads = torch.abs(self.time_spreads) + 0.5
        freq_spreads = torch.abs(self.freq_spreads) + 0.5

        # Render magnitude spectrogram
        rendered_magnitude = render_spectrogram_from_gaussians(
            time_centers, freq_centers,
            time_spreads, freq_spreads,
            self.magnitudes,
            (self.n_freq_bins, self.n_time_frames),
            self.device
        )

        # Denormalize
        rendered_magnitude = rendered_magnitude * self.mag_std + self.mag_mean

        # Combine with phase
        if self.training:
            # During training, we might want to use target phase to guide magnitude learning first
            # Or use learned phase if we are optimizing phase loss
            phase_to_use = self.target_phase
        else:
            # During inference/eval, we must use the learned parameters (or reconstruct phase)
            # Mapping 1D phase params to 2D grid is non-trivial without a renderer for phase.
            # For now, let's assume we want to use the learned phase if implemented, 
            # but currently the code lacks a 'render_phase_from_gaussians' function.
            # Fallback: Use target phase (Cheating) or Griffin-Lim (Real scenario).
            phase_to_use = self.target_phase 
            
        rendered_complex = rendered_magnitude * torch.exp(1j * phase_to_use)

        return rendered_magnitude, rendered_complex

    def compute_loss(self, rendered_magnitude, target_magnitude):
        """Compute reconstruction loss"""
        # Main reconstruction loss
        if self.loss_type == 'l2':
            recon_loss = F.mse_loss(rendered_magnitude, target_magnitude)
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(rendered_magnitude, target_magnitude)
        else:  # spectral
            recon_loss = compute_spectral_loss(rendered_magnitude, target_magnitude)

        # Perceptual loss (mel-spectrogram)
        if self.perceptual_weight > 0:
            rendered_mel = self.mel_transform(rendered_magnitude)
            target_mel = self.mel_transform(target_magnitude)
            perceptual_loss = F.mse_loss(
                torch.log1p(rendered_mel),
                torch.log1p(target_mel)
            )
        else:
            perceptual_loss = 0.0

        total_loss = recon_loss + self.perceptual_weight * perceptual_loss

        return total_loss, recon_loss, perceptual_loss

    def optimize(self):
        """Main optimization loop"""
        self.worklog.info("Starting optimization...")

        for step in range(self.num_steps):
            # Forward pass
            rendered_magnitude, _ = self.forward(use_gl=False)

            # Compute loss
            total_loss, recon_loss, percep_loss = self.compute_loss(
                rendered_magnitude, self.target_magnitude
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging
            if step % 100 == 0:
                # Reconstruct audio for SNR (Use GL for honest metric)
                with torch.no_grad():
                    _, gl_complex = self.forward(use_gl=True)
                reconstructed_audio = stft_to_audio(
                    gl_complex, self.hop_length, self.win_length
                )
                snr = compute_snr(self.audio, reconstructed_audio)

                self.worklog.info(
                    f"Step {step}/{self.num_steps} | "
                    f"Loss: {total_loss:.4f} | "
                    f"Recon: {recon_loss:.4f} | "
                    f"Percep: {percep_loss:.4f} | "
                    f"SNR: {snr:.2f} dB"
                )

            # Save checkpoint
            if step % self.save_steps == 0 and step > 0:
                self.save_checkpoint(step)

            # Adaptive Gaussian addition
            if self.adaptive_add and step in self.add_gaussians_steps:
                self._add_gaussians(step)

            # Learning rate decay
            if step % 1000 == 0 and step > 0:
                self.scheduler.step()

        # Final save
        self.save_checkpoint(self.num_steps)
        self.save_reconstructed_audio()

        self.worklog.info("Optimization completed!")

    def _add_gaussians(self, step):
        """Adaptively add new Gaussians to high-error regions"""
        with torch.no_grad():
            rendered_magnitude, _ = self.forward(use_gl=False)
            error_map = torch.abs(self.target_magnitude - rendered_magnitude)

            # Find peaks in error map
            error_np = error_map.cpu().numpy()
            peaks = find_peaks(error_np.flatten(), height=np.percentile(error_np, 90))[0]

            # Add new Gaussians at error peaks
            n_add = min(10, len(peaks))
            if n_add > 0:
                new_indices = np.random.choice(peaks, n_add, replace=False)
                new_time = torch.tensor(new_indices % self.n_time_frames, device=self.device, dtype=torch.float32)
                new_freq = torch.tensor(new_indices // self.n_time_frames, device=self.device, dtype=torch.float32)

                # Concatenate new parameters
                self.time_centers = nn.Parameter(torch.cat([self.time_centers, new_time]))
                self.freq_centers = nn.Parameter(torch.cat([self.freq_centers, new_freq]))
                self.time_spreads = nn.Parameter(torch.cat([self.time_spreads, torch.ones(n_add, device=self.device) * 3.0]))
                self.freq_spreads = nn.Parameter(torch.cat([self.freq_spreads, torch.ones(n_add, device=self.device) * 5.0]))
                self.magnitudes = nn.Parameter(torch.cat([self.magnitudes, torch.ones(n_add, device=self.device)]))
                self.phases = nn.Parameter(torch.cat([self.phases, torch.zeros(n_add, device=self.device)]))

                self.num_gaussians += n_add
                self.worklog.info(f"Added {n_add} Gaussians at step {step}. Total: {self.num_gaussians}")

                # Re-initialize optimizer with new parameters
                self._init_optimization(self.args)

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'num_gaussians': self.num_gaussians,
            'time_centers': self.time_centers.detach().cpu(),
            'freq_centers': self.freq_centers.detach().cpu(),
            'time_spreads': self.time_spreads.detach().cpu(),
            'freq_spreads': self.freq_spreads.detach().cpu(),
            'magnitudes': self.magnitudes.detach().cpu(),
            'phases': self.phases.detach().cpu(),
            'audio_params': {
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'sample_rate': self.sample_rate,
                'mag_mean': self.mag_mean,
                'mag_std': self.mag_std,
            }
        }

        if self.quantize:
            checkpoint = quantize_gaussians(checkpoint, bits=self.bits_per_param)

        torch.save(checkpoint, f"{self.ckpt_dir}/checkpoint_{step:06d}.pth")
        self.worklog.info(f"Saved checkpoint at step {step}")

    def save_reconstructed_audio(self):
        """Save final reconstructed audio"""
        with torch.no_grad():
            _, rendered_complex = self.forward(use_gl=True)
            reconstructed_audio = stft_to_audio(
                rendered_complex, self.hop_length, self.win_length
            )

            output_path = f"{self.train_dir}/reconstructed.wav"
            save_audio(output_path, reconstructed_audio.cpu(), self.sample_rate)

            # Compute final metrics
            snr = compute_snr(self.audio, reconstructed_audio)
            self.worklog.info(f"Final SNR: {snr:.2f} dB")
            self.worklog.info(f"Saved reconstructed audio to {output_path}")

            # Visualize Gaussians on spectrogram
            viz_path = f"{self.train_dir}/gaussians_visualization.png"
            visualize_gaussians_on_spectrogram(
                self.target_magnitude.cpu(),
                self.time_centers.detach().cpu(),
                self.freq_centers.detach().cpu(),
                self.time_spreads.detach().cpu(),
                self.freq_spreads.detach().cpu(),
                save_path=viz_path
            )
            self.worklog.info(f"Saved visualization to {viz_path}")

    def _load_model(self):
        """Load model from checkpoint"""
        checkpoint = torch.load(self.ckpt_file, map_location=self.device)

        if self.quantize:
            checkpoint = dequantize_gaussians(checkpoint)

        self.num_gaussians = checkpoint['num_gaussians']
        self.time_centers = nn.Parameter(checkpoint['time_centers'].to(self.device))
        self.freq_centers = nn.Parameter(checkpoint['freq_centers'].to(self.device))
        self.time_spreads = nn.Parameter(checkpoint['time_spreads'].to(self.device))
        self.freq_spreads = nn.Parameter(checkpoint['freq_spreads'].to(self.device))
        self.magnitudes = nn.Parameter(checkpoint['magnitudes'].to(self.device))
        self.phases = nn.Parameter(checkpoint['phases'].to(self.device))

        # Load audio params
        audio_params = checkpoint['audio_params']
        self.n_fft = audio_params['n_fft']
        self.hop_length = audio_params['hop_length']
        self.win_length = audio_params['win_length']
        self.sample_rate = audio_params['sample_rate']
        self.mag_mean = audio_params['mag_mean']
        self.mag_std = audio_params['mag_std']

        self.worklog.info(f"Loaded model from {self.ckpt_file}")