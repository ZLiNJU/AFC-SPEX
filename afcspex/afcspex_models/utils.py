import torch
import torch.nn as nn

def get_complex_masks_from_stacked(real_mask: torch.Tensor):
    """
    Construct the complex clean speech and noise mask from the estimated stacked clean speech mask. Inverts the
    compression by tanh output activation to the range [-inf, inf] for real and imaginary components.

    :param real_mask: estimated mask with stacked real and imaginary components [BATCH, 2, F, T]
    :return: the complex masks [B, F, T]
    """
    cirm_K = 1
    cirm_C = 1
    compressed_complex_speech_mask = real_mask[:, 0, ...] + (1j) * real_mask[:, 1, ...]
    complex_speech_mask = (-1 / cirm_C) * torch.log(
        (cirm_K - cirm_K * compressed_complex_speech_mask) / (
                cirm_K + cirm_K * compressed_complex_speech_mask))
    complex_noise_mask = (1 - torch.real(complex_speech_mask)) - (1j) * torch.imag(complex_speech_mask)
    return complex_speech_mask, complex_noise_mask

class PartitionedFFT(nn.Module):
    '''
    Partitioned FFT and IFFT implementation.
    '''
    def __init__(self, fft_len: int = 512, hop_len: int = 256):
        '''
        :param  fft_len (int): Length of the FFT. Must be even and divisible by hop_len.
        :param  hop_len (int): Hop length between consecutive frames.
        '''
        if fft_len % 2 != 0:
            raise ValueError(f"fft_len must be even, but get value: {fft_len}.")
        if fft_len % hop_len != 0:
            raise ValueError(f"fft_len ({fft_len}) must be divisible by hop_len ({hop_len}).")
        
        super().__init__()
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.n_overlap = fft_len // hop_len

    def forward(self, td_signal: torch.Tensor = None, fd_signal: torch.Tensor = None):
        """If the input is time-domain signal, provide v and set V=None, implementing FFT;
        if the input is frequency-domain signal, provide V and set v=None, implementing IFFT

        :param td_signal: Time-domain signal of shape (n_batch, time, n_mic).
        :type td_signal: torch.Tensor, optional
        :param fd_signal: Frequency-domain signal of shape (n_batch, n_bin, n_frame, n_mic).
        :type fd_signal: torch.Tensor, optional
        :return: The transformed signal.
                 If FFT is performed, returns the frequency-domain signal `fd_out` of shape (n_batch, n_bin, n_frame, n_mic).
                 If IFFT is performed, returns the time-domain signal `td_out` of shape (n_batch, time, n_mic).
        :rtype: torch.Tensor
        """
        if fd_signal is None and td_signal is not None:
            device = td_signal.device
            shape_v = td_signal.shape # (n_batch, time, n_mic)
            n_mic = shape_v[2]
            n_time = shape_v[1]
            n_batch = shape_v[0]
            n_frame = (n_time + self.hop_len - 1) // self.hop_len
            v = torch.cat([td_signal, torch.zeros((n_batch, self.hop_len * n_frame - n_time, n_mic), 
                                                  device=device)], dim=1)
            v_t = torch.transpose(v.reshape(n_batch, -1, self.hop_len, n_mic), 1, 2)  # (n_batch, hop_len, n_frame, n_mic)
            vt0 = torch.cat([torch.zeros((n_batch, (self.n_overlap - 1) * self.hop_len, n_frame, n_mic), \
                                         device=device), v_t], dim=1) # (n_batch, fft_len, n_frame, n_mic)
            fd_out = torch.fft.rfft(vt0, self.fft_len, dim=1)  # (n_batch, n_bin, n_frame, n_mic)
            return fd_out

        elif td_signal is None and fd_signal is not None:
            n_mic = fd_signal.shape[3]
            n_batch = fd_signal.shape[0]
            v_tf = torch.fft.irfft(fd_signal, self.fft_len, dim=1)
            v_t1 = v_tf[:, (self.n_overlap - 1) * self.hop_len:, :, :] # (n_batch, hop_len, n_frame, n_mic)
            td_out = torch.transpose(v_t1, 1, 2).reshape(n_batch, -1, n_mic)
            return td_out

        else:
            raise ValueError("Invalid input: either tf_signal or fd_signal must be provided, not both.")