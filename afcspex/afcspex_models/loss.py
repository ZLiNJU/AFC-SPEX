import torch
import torch.nn as nn
from afcspex_models.utils import PartitionedFFT

def cal_loss(clean_td, est_clean_td, noise_td, est_noise_td,
            clean_stft, est_clean_stft, noise_stft, est_noise_stft):
    """
    Compute the loss based on L1-norms of time domain speech and noise signals and frequency magnitudes.

    :param clean_td: target clean signal in time domain
    :param est_clean_td: estimated clean signal in time domain
    :param noise_td: target noise signal in time domain
    :param est_noise_td: estimated noise signal in time domain
    :param clean_stft: target clean signal in STFT domain
    :param est_clean_stft: estimated clean signal in STFT domain
    :param noise_stft: target noise signal in STFT domain
    :param est_noise_stft: estimated noise signal in STFT domain
    :return: four loss terms based on L1-loss
    """
    clean_td_loss = torch.mean(torch.abs(clean_td - est_clean_td))
    noise_td_loss = torch.mean(torch.abs(noise_td - est_noise_td))
    clean_mag_loss = torch.mean(torch.abs(torch.abs(clean_stft) - torch.abs(est_clean_stft)))
    noise_mag_loss = torch.mean(torch.abs(torch.abs(noise_stft) - torch.abs(est_noise_stft)))

    return clean_td_loss, noise_td_loss, clean_mag_loss, noise_mag_loss

class TimeFreqLoss(nn.Module):
    def __init__(self, fft_len: int = 512, hop_len: int = 256):
        super().__init__()
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.ft = PartitionedFFT(fft_len=self.fft_len, hop_len=self.hop_len)

    def forward(self, clean: torch.Tensor, enhanced: torch.Tensor, fd_enhanced: torch.Tensor,
                noise: torch.Tensor, est_noise: torch.Tensor, fd_est_noise: torch.Tensor):
        """
        Loss function combing time domain and frequency domain error.
        :param clean: target clean signal in time domain [n_batch, n_time]
        :param enhanced: estimated clean signal in time domain [n_batch, n_time]
        :param fd_enhanced: estimated clean signal in frequency domain [n_batch, n_bin, n_frame]
        :param noise: real noise signal in time domain [n_batch, n_time]
        :param est_noise: estimated noise signal in time domain [n_batch, n_time]
        :param fd_est_noise: estimated noise signal in frequency domain [n_batch, n_bin, n_frame]

        :return: Loss value
        """
        fd_clean = self.ft(td_signal=clean[:, :, None], fd_signal=None)  # [n_batch, n_time] -> [n_batch, n_bin, n_frame, n_mic=1]
        fd_clean = fd_clean.squeeze(-1) # [n_batch, n_bin, n_frame]
        fd_noise = self.ft(td_signal=noise[:, :, None], fd_signal=None)  # [n_batch, n_time] -> [n_batch, n_bin, n_frame, n_mic=1]
        fd_noise = fd_noise.squeeze(-1) # [n_batch, n_bin, n_frame]

        clean_td_loss, noise_td_loss, clean_mag_loss, noise_mag_loss = cal_loss(
            clean_td=clean, est_clean_td=enhanced,
            noise_td=noise, est_noise_td=est_noise,
            clean_stft=fd_clean, est_clean_stft=fd_enhanced,
            noise_stft=fd_noise, est_noise_stft=fd_est_noise
        )

        loss = 10 * (clean_td_loss + noise_td_loss) + (clean_mag_loss + noise_mag_loss)

        return loss
